import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import List, Dict, Optional
from omegaconf import DictConfig
from .mamba2_minimal import Mamba2, Mamba2Config
from .external_depth import ExternalDepthModel


class ResidualConvUnit(nn.Module):
    """Residual convolutional unit from RefineNet."""

    # ... (no changes to this class)

    def __init__(
        self,
        features: int,
        use_bn: bool = False,
        activation: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, padding=1, bias=not use_bn
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, padding=1, bias=not use_bn
        )

        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        return x + out


class FeatureFusionBlock(nn.Module):
    """Feature fusion block for combining multi-scale features."""

    def __init__(
        self,
        features: int,
        skip_channels: int = 0,
        use_bn: bool = False,
        activation: nn.Module = nn.ReLU(inplace=True),
        deconv: bool = False,
        expand: bool = False,
        upsample: bool = True,
    ):
        super().__init__()

        out_features = features
        if expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, bias=True)

        self.resConfUnit1 = ResidualConvUnit(features, use_bn, activation)
        self.resConfUnit2 = ResidualConvUnit(features, use_bn, activation)

        # Projection for skip connection if channels don't match
        self.skip_proj = nn.Identity()
        if skip_channels != 0 and skip_channels != features:
            self.skip_proj = nn.Conv2d(skip_channels, features, kernel_size=1)

        if upsample:
            if deconv:
                self.upsample = nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=2, stride=2
                )
            else:
                self.upsample = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )
        else:
            self.upsample = nn.Identity()

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        output = xs[0]

        if len(xs) == 2:
            skip = xs[1]

            # Project skip connection if needed
            skip = self.skip_proj(skip)

            # Resize second input to match first
            res = nn.functional.interpolate(
                skip,
                size=(output.shape[2], output.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
            output = output + res

        output = self.resConfUnit1(output)
        output = self.resConfUnit2(output)
        output = self.out_conv(output)
        output = self.upsample(output)

        return output


class ReassembleBlock(nn.Module):
    """Reassemble tokens into image-like representation."""

    # ... (no changes)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        read_operation: str = "project",
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.read_operation = read_operation

        # Read operation for handling CLS/readout token
        if read_operation == "project":
            self.read_proj = nn.Sequential(
                nn.Linear(2 * in_channels, in_channels), nn.GELU()
            )

        # Project to output channels (1x1 Conv - pure projection)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        tokens: torch.Tensor,
        patch_size: int,
        image_size: tuple[int, int] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, N+1, C) where N is number of patches, +1 for CLS token
            patch_size: Size of each patch in the original image
            image_size: Tuple of (H, W) of the original image
        """
        B, T, C = tokens.shape

        # Calculate grid size first to know how many tokens are spatial patches
        if image_size is not None:
            H_img, W_img = image_size
            H = H_img // patch_size
            W = W_img // patch_size
        else:
            # Fallback estimation if image_size not provided (assumes no registers)
            N_spatial = T - 1
            H = W = int(N_spatial**0.5)

        N_spatial = H * W

        # Separate CLS token and patch tokens
        # We assume CLS is at index 0, and patches are the LAST N_spatial tokens
        cls_token = tokens[:, 0:1, :]  # (B, 1, C)
        patch_tokens = tokens[:, -N_spatial:, :]  # (B, N_spatial, C)

        # Apply read operation
        if self.read_operation == "ignore":
            tokens_out = patch_tokens
        elif self.read_operation == "add":
            tokens_out = patch_tokens + cls_token
        elif self.read_operation == "project":
            # Concatenate cls_token to each patch token and project
            cls_expanded = cls_token.expand(-1, patch_tokens.shape[1], -1)
            tokens_cat = torch.cat([patch_tokens, cls_expanded], dim=-1)
            tokens_out = self.read_proj(tokens_cat)
        else:
            raise ValueError(f"Unknown read operation: {self.read_operation}")

        assert (
            H * W == tokens_out.shape[1]
        ), f"Token count {tokens_out.shape[1]} does not match grid {H}x{W}"

        tokens_2d = tokens_out.transpose(1, 2).reshape(B, tokens_out.shape[-1], H, W)

        # Project
        out = self.proj(tokens_2d)

        # Resample
        if self.scale_factor != 1.0:
            out = F.interpolate(
                out, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
            )

        return out


class DPTHead(nn.Module):
    """DPT head: channel projection only, no spatial upsampling."""

    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthHead(DPTHead):
    """Depth prediction head outputting HxWx1."""

    def __init__(self, in_channels: int, use_bn: bool = False):
        super().__init__(in_channels, out_channels=1, use_bn=use_bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        # 1 & 2: Predict inverse depth and lower bound
        # Increase eps to 0.1 to cap max depth at 10.0 (prevents infinite depth cheat)
        eps = 0.1
        disp = F.softplus(out) + eps
        depth = 1.0 / disp
        return depth


class MotionHead(DPTHead):
    """Motion prediction head outputting HxWx6 (linear and angular velocity)."""

    def __init__(self, in_channels: int, use_bn: bool = False):
        super().__init__(in_channels, out_channels=6, use_bn=use_bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        motion = super().forward(x)
        return motion


def se3_exp_map(
    v: torch.Tensor, omega: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # ... (no changes)
    """
    Compute SE(3) exponential map: exp(t * ξ̂) where ξ = (v, ω)
    """
    # Handle both (B, 3, H, W) and (B, N, H, W, 3) formats
    if v.dim() == 4:  # (B, 3, H, W)
        v = v.permute(0, 2, 3, 1)  # (B, H, W, 3)
        omega = omega.permute(0, 2, 3, 1)  # (B, H, W, 3)
        has_time_dim = False
    else:  # (B, N, H, W, 3)
        has_time_dim = True

    device = v.device

    # Scale by time - handle different t shapes
    if isinstance(t, (int, float)):
        omega_t = omega * t
        v_t = v * t
    elif t.dim() == 1:  # (N,)
        # Expand t to match omega/v dimensions
        if has_time_dim:
            t = t.view(1, -1, 1, 1, 1)  # (1, N, 1, 1, 1)
        else:
            pass
        omega_t = omega * t
        v_t = v * t
    else:  # Already shaped correctly
        omega_t = omega * t
        v_t = v * t

    # Compute rotation angle (theta)
    theta = torch.norm(omega_t, dim=-1, keepdim=True)  # (..., 1)
    eps = 1e-6

    # Normalized rotation axis
    omega_normalized = omega_t / (theta + eps)

    # Create identity matrix - match dimensions
    shape = omega_normalized.shape[:-1]  # All dims except last
    I = (
        torch.eye(3, device=device)
        .view(*([1] * len(shape)), 3, 3)
        .expand(*shape, -1, -1)
    )

    # Skew-symmetric matrix K
    wx, wy, wz = (
        omega_normalized[..., 0:1],
        omega_normalized[..., 1:2],
        omega_normalized[..., 2:3],
    )
    zeros = torch.zeros_like(wx)

    K = torch.stack(
        [
            torch.cat([zeros, -wz, wy], dim=-1),
            torch.cat([wz, zeros, -wx], dim=-1),
            torch.cat([-wy, wx, zeros], dim=-1),
        ],
        dim=-2,
    )

    # K²
    K2 = torch.matmul(K, K)

    # Rodrigues formula
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)

    R = I + sin_theta * K + (1 - cos_theta) * K2

    # Small angle approximation
    small_angle_mask = (theta < eps).unsqueeze(-1)
    R = torch.where(small_angle_mask, I + K * theta.unsqueeze(-1), R)

    T = v_t

    return R, T


def motion_blur(
    sharp_image: torch.Tensor,
    depth: torch.Tensor,
    motion: torch.Tensor,
    camera_matrix: torch.Tensor = None,
    num_samples: int = 16,
    exposure_time: float = 1.0,
) -> torch.Tensor:
    # ... (no changes)
    """
    Apply motion blur using vectorized SE(3) integration.
    """
    B, C, H, W = sharp_image.shape
    device = sharp_image.device

    v = motion[:, 0:3, :, :]  # (B, 3, H, W)
    omega = motion[:, 3:6, :, :]  # (B, 3, H, W)

    # Default camera matrix
    if camera_matrix is None:
        fx = fy = W / 2.0
        cx = W / 2.0
        cy = H / 2.0
        camera_matrix = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32
        )

    if camera_matrix.dim() == 2:
        camera_matrix = camera_matrix.unsqueeze(0).expand(B, -1, -1)

    K_inv = torch.inverse(camera_matrix)  # (B, 3, 3)

    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(x_coords)
    pixel_coords = torch.stack([x_coords, y_coords, ones], dim=-1)  # (H, W, 3)
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)

    # Back-project to 3D
    depth_expanded = depth.squeeze(1).unsqueeze(-1)  # (B, H, W, 1)
    pixel_coords_expanded = pixel_coords.unsqueeze(-1)  # (B, H, W, 3, 1)
    K_inv_expanded = K_inv.view(B, 1, 1, 3, 3).expand(-1, H, W, -1, -1)
    camera_coords = torch.matmul(K_inv_expanded, pixel_coords_expanded).squeeze(-1)
    P = depth_expanded * camera_coords  # (B, H, W, 3)

    # ===== VECTORIZED TIME SAMPLING =====
    # Sample all time points at once
    t_samples = torch.linspace(0, exposure_time, num_samples, device=device)  # (N,)
    N = num_samples

    # Expand velocities for all time samples: (B, H, W, 3) -> (B, N, H, W, 3)
    v_expanded = v.permute(0, 2, 3, 1).unsqueeze(1).expand(-1, N, -1, -1, -1)
    omega_expanded = omega.permute(0, 2, 3, 1).unsqueeze(1).expand(-1, N, -1, -1, -1)

    # Expand t_samples: (N,) -> (B, N, H, W, 1)
    t_expanded = t_samples.view(1, N, 1, 1, 1).expand(B, -1, H, W, -1)

    # Compute SE(3) for all time samples at once
    R, T = se3_exp_map(v_expanded, omega_expanded, t_expanded)

    # Expand P for all time samples: (B, H, W, 3) -> (B, N, H, W, 3)
    P_expanded = P.unsqueeze(1).expand(-1, N, -1, -1, -1)

    # Apply SE(3) transformation for all samples: P_t = R @ P + T
    P_t = torch.matmul(R, P_expanded.unsqueeze(-1)).squeeze(-1) + T  # (B, N, H, W, 3)

    # Project back to image plane for all samples
    K_expanded = camera_matrix.view(B, 1, 1, 1, 3, 3).expand(-1, N, H, W, -1, -1)
    p_homogeneous = torch.matmul(K_expanded, P_t.unsqueeze(-1)).squeeze(
        -1
    )  # (B, N, H, W, 3)

    # Normalize to get pixel coordinates
    eps = 1e-6
    z = p_homogeneous[..., 2:3]  # (B, N, H, W, 1)
    x_proj = p_homogeneous[..., 0:1] / (z + eps)
    y_proj = p_homogeneous[..., 1:2] / (z + eps)

    # Normalize to [-1, 1] for grid_sample
    x_norm = 2.0 * x_proj / (W - 1) - 1.0
    y_norm = 2.0 * y_proj / (H - 1) - 1.0
    grid = torch.cat([x_norm, y_norm], dim=-1)  # (B, N, H, W, 2)

    # Reshape for grid_sample: (B*N, H, W, 2)
    grid_flat = grid.reshape(B * N, H, W, 2)

    # Expand sharp_image for all time samples: (B, C, H, W) -> (B*N, C, H, W)
    sharp_image_expanded = (
        sharp_image.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)
    )

    # Sample all at once
    sampled_flat = F.grid_sample(
        sharp_image_expanded,
        grid_flat,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B*N, C, H, W)

    # Reshape back and average: (B*N, C, H, W) -> (B, N, C, H, W) -> (B, C, H, W)
    sampled = sampled_flat.reshape(B, N, C, H, W)
    blurred_image = sampled.mean(dim=1)  # Average over time samples

    return blurred_image


# --- New Architecture Classes ---


class FusionBlock(nn.Module):
    # ... (no changes)
    """
    Fusion block for combining RGB, Depth, and Motion features.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, f_rgb: torch.Tensor, f_depth: torch.Tensor, f_motion: torch.Tensor
    ) -> torch.Tensor:
        f = torch.cat([f_rgb, f_depth, f_motion], dim=1)
        return self.conv(f)


class SimpleEncoder(nn.Module):
    # ... (no changes)
    """
    Lightweight encoder for specific modality.
    """

    def __init__(self, in_channels: int, base_channels: int, num_stages: int = 5):
        super().__init__()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        curr_channels = base_channels
        for i in range(num_stages - 1):  # -1 because stem is stage 0
            # Downsample
            self.downsamples.append(
                nn.Conv2d(curr_channels, curr_channels * 2, 3, stride=2, padding=1)
            )
            curr_channels *= 2
            # Conv block
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(curr_channels, curr_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(curr_channels, curr_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.stem(x)
        features.append(x)  # Scale 0

        for down, stage in zip(self.downsamples, self.stages):
            x = down(x)
            x = stage(x)
            features.append(x)  # Scale i+1

        return features


class SSMBottleneck(nn.Module):
    # ... (no changes)
    """
    Bottleneck using Mamba2 SSM with 4-way directional scanning.
    """

    def __init__(
        self, channels: int, num_layers: int = 2, d_state: int = 64, expand: int = 2
    ):
        super().__init__()
        self.channels = channels

        self.config = Mamba2Config(
            d_model=channels,
            n_layer=num_layers,
            d_state=d_state,
            expand=expand,
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {"norm": nn.LayerNorm(channels), "mixer": Mamba2(self.config)}
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Flatten: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)

        # Padding for chunk_size
        seq_len = x_flat.shape[1]
        chunk_size = self.config.chunk_size
        pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size

        if pad_len > 0:
            x_flat = F.pad(x_flat, (0, 0, 0, pad_len))

        residual = x_flat

        for layer in self.layers:
            norm_x = layer["norm"](residual)
            mixer = layer["mixer"]

            # 1. Forward scan
            out_f, _ = mixer(norm_x)

            # 2. Backward scan
            out_b, _ = mixer(norm_x.flip([1]))
            out_b = out_b.flip([1])

            # Prepare Transposed inputs
            # Unpad current norm_x to get valid 2D spatial map
            valid_x = norm_x[:, :seq_len, :]
            img_x = valid_x.transpose(1, 2).reshape(B, C, H, W)

            # Transpose (Swap H and W)
            img_t = img_x.transpose(2, 3)  # (B, C, W, H)
            flat_t = img_t.flatten(2).transpose(1, 2)  # (B, W*H, C)

            if pad_len > 0:
                flat_t_padded = F.pad(flat_t, (0, 0, 0, pad_len))
            else:
                flat_t_padded = flat_t

            # 3. Transposed Forward
            out_tf, _ = mixer(flat_t_padded)

            # 4. Transposed Backward
            out_tb, _ = mixer(flat_t_padded.flip([1]))
            out_tb = out_tb.flip([1])

            # Unpad and reshape transposed outputs back to original domain
            out_tf = out_tf[:, :seq_len, :]
            out_tb = out_tb[:, :seq_len, :]

            out_tf_img = out_tf.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)
            out_tf_final = out_tf_img.flatten(2).transpose(1, 2)

            out_tb_img = out_tb.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)
            out_tb_final = out_tb_img.flatten(2).transpose(1, 2)

            # Sum aggregation (using unpadded valid outputs for 1 & 2)
            out_f_valid = out_f[:, :seq_len, :]
            out_b_valid = out_b[:, :seq_len, :]

            aggregated = out_f_valid + out_b_valid + out_tf_final + out_tb_final

            # Residual update
            valid_residual = residual[:, :seq_len, :] + aggregated

            if pad_len > 0:
                residual = F.pad(valid_residual, (0, 0, 0, pad_len))
            else:
                residual = valid_residual

        # Final output
        final_out = residual[:, :seq_len, :]
        return final_out.transpose(1, 2).reshape(B, C, H, W)


class RefinementUNet(nn.Module):
    # ... (no changes)
    """
    U-Net for image refinement using RGB, Depth, and Motion.
    Strategy 1: Multi-encoder + feature-level fusion.
    """

    def __init__(self, base_channels: int = 32, ssm_cfg: Optional[DictConfig] = None):
        super().__init__()

        # 3 Encoders (RGB, Depth, Motion)
        # We need 5 scales to reach H/16: 0(H), 1(H/2), 2(H/4), 3(H/8), 4(H/16)
        num_stages = 5
        self.enc_rgb = SimpleEncoder(3, base_channels, num_stages)
        self.enc_depth = SimpleEncoder(1, base_channels, num_stages)
        self.enc_motion = SimpleEncoder(
            6, base_channels, num_stages
        )  # 6 channels for motion

        # Fusion Blocks at each scale
        self.fusion_blocks = nn.ModuleList()
        curr_channels = base_channels
        channels_list = []
        for _ in range(num_stages):
            self.fusion_blocks.append(FusionBlock(curr_channels))
            channels_list.append(curr_channels)
            curr_channels *= 2

        # Bottleneck (at the smallest resolution, features from last fusion)
        last_channels = channels_list[-1]

        if ssm_cfg is not None:
            self.bottleneck = SSMBottleneck(
                channels=last_channels,
                num_layers=ssm_cfg.n_layer,
                d_state=ssm_cfg.d_state,
                expand=ssm_cfg.expand,
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(last_channels, last_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_channels, last_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        # Decoder
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Iterate backwards from second to last scale
        # Scales: 0, 1, 2, 3, 4. Bottleneck is at 4.
        # Decoder will go 4->3, 3->2, 2->1, 1->0

        d_channels = last_channels
        for i in range(num_stages - 2, -1, -1):  # 3, 2, 1, 0
            # Upsample from d_channels to channels_list[i]
            target_channels = channels_list[i]

            self.decoder_upsamples.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )

            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        d_channels + target_channels, target_channels, 3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(target_channels, target_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            d_channels = target_channels  # Update for next iteration

        # Final head to predict residual
        self.head = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor, motion: torch.Tensor
    ) -> torch.Tensor:
        # Encode
        feats_rgb = self.enc_rgb(rgb)
        feats_depth = self.enc_depth(depth)
        feats_motion = self.enc_motion(motion)

        # Fuse
        fused_feats = []
        for i in range(len(feats_rgb)):
            fused = self.fusion_blocks[i](feats_rgb[i], feats_depth[i], feats_motion[i])
            fused_feats.append(fused)

        # Bottleneck
        x = self.bottleneck(fused_feats[-1])

        # Decode
        # We start from the bottleneck output which corresponds to the last scale
        # And we skip-connect the fused features from previous scales

        for i, (up, block) in enumerate(
            zip(self.decoder_upsamples, self.decoder_blocks)
        ):
            # Index of skip connection: (N-2) - i
            # If N=5, indices are 4 (bottleneck), 3, 2, 1, 0
            # i=0: skip index 3
            skip_idx = (len(fused_feats) - 2) - i
            skip = fused_feats[skip_idx]

            x = up(x)

            # Handle potential size mismatch due to odd dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                )

            x = torch.cat([x, skip], dim=1)
            x = block(x)

        # Prediction
        residual = self.head(x)
        return torch.clamp(rgb + residual, 0.0, 1.0)


class DPT(nn.Module):
    """
    Dense Prediction Transformer with configurable backbone and separate Fusion paths,
    plus a Refinement U-Net.
    Refactored to rely exclusively on External Depth Prior (Scale-Invariant).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Load backbone (frozen or trainable based on config)
        self.backbone = AutoModel.from_pretrained(
            cfg.backbone.name, trust_remote_code=True, output_hidden_states=True
        )

        if cfg.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get backbone configuration
        self.backbone_config = self.backbone.config
        self.embed_dim = getattr(
            self.backbone_config, "hidden_size", cfg.backbone.embed_dim
        )

        # Determine which layers to tap for skip connections
        self.feature_layers = cfg.decoder.feature_layers

        # Reassemble blocks for each feature layer
        self.reassemble_blocks = nn.ModuleList()
        scales = cfg.decoder.scales  # e.g., [4, 8, 16, 32]

        for i, (layer_idx, scale) in enumerate(zip(self.feature_layers, scales)):
            self.reassemble_blocks.append(
                ReassembleBlock(
                    in_channels=self.embed_dim,
                    out_channels=cfg.decoder.features,
                    scale_factor=scale / cfg.backbone.patch_size,
                    read_operation=cfg.decoder.read_operation,
                )
            )

        checkpoint = self.depth_prior_cfg.checkpoint
        freeze_backbone = self.depth_prior_cfg.freeze_backbone
        fine_tune_head = self.depth_prior_cfg.fine_tune_head
        default_focal = getattr(self.depth_prior_cfg, "default_focal_length", 700.0)

        self.external_depth = ExternalDepthModel(
            checkpoint=checkpoint,
            freeze_backbone=freeze_backbone,
            fine_tune_head=fine_tune_head,
            default_focal_length=default_focal,
        )

        # Motion Fusion Path (Internal Depth path removed)
        self.motion_fusion_blocks = nn.ModuleList()

        num_layers = len(self.feature_layers)
        fusion_channels = getattr(cfg.decoder, "fusion_channels", [64, 128, 256, 256])

        assert (
            len(fusion_channels) == num_layers
        ), f"Number of fusion channel widths ({len(fusion_channels)}) must match number of layers ({num_layers})"

        # Initialize Fusion Blocks for Motion path
        for i in range(num_layers):
            expand = i != num_layers - 1
            f = fusion_channels[i]
            skip_ch = cfg.decoder.features if i > 0 else 0

            self.motion_fusion_blocks.append(
                FeatureFusionBlock(
                    features=f,
                    skip_channels=skip_ch,
                    use_bn=cfg.decoder.use_bn,
                    deconv=cfg.decoder.use_deconv,
                    expand=expand,
                    upsample=(i != 0),
                )
            )

        # Prediction heads (Only Motion)
        final_features = cfg.decoder.features // (2 ** (num_layers - 1))

        self.motion_head = MotionHead(
            in_channels=final_features, use_bn=cfg.heads.use_bn
        )

        # Refinement U-Net (Strategy 1)
        # Using a default base_channels of 32 for "lightweight" encoders
        ssm_cfg = getattr(cfg, "ssm", None)
        self.refinement_net = RefinementUNet(base_channels=32, ssm_cfg=ssm_cfg)

    def forward_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from backbone at multiple layers."""
        outputs = self.backbone(x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        features = []
        for layer_idx in self.feature_layers:
            features.append(hidden_states[layer_idx])
        return features

    def forward(
        self, x: torch.Tensor, gt_sharp: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, H, W)
        """
        # Pad input to be divisible by patch_size (16)
        H, W = x.shape[2], x.shape[3]
        patch_size = self.cfg.backbone.patch_size

        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        x_padded = x
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Extract multi-scale features from backbone
        backbone_features = self.forward_backbone(x_padded)

        # Reassemble features
        reassembled = []
        for feat, reassemble_block in zip(backbone_features, self.reassemble_blocks):
            reassembled.append(
                reassemble_block(
                    feat,
                    self.cfg.backbone.patch_size,
                    image_size=(x_padded.shape[2], x_padded.shape[3]),
                )
            )

        # --- Depth Path (External Scale-Invariant) ---
        # 1. Get raw depth from External Model
        ext_out = self.external_depth(x)
        depth_raw = ext_out["depth"]  # (B, 1, H, W)

        # 2. Scale-Invariant Normalization (Option A)
        # Ensure positive
        depth = F.softplus(depth_raw) + 1e-3
        # Normalize mean to 1.0 per image
        # This resolves the scale ambiguity: Network learns velocity relative to mean depth = 1
        mean_depth = depth.mean(dim=(2, 3), keepdim=True)
        depth = depth / (mean_depth + 1e-6)

        # Extract intrinsics if available
        camera_matrix = None
        if self.depth_prior_cfg and self.depth_prior_cfg.get(
            "use_predicted_intrinsics", False
        ):
            if "intrinsics" in ext_out:
                camera_matrix = ext_out["intrinsics"]

        # --- Motion Path ---
        fused_motion = reassembled[-1]
        for i in range(len(reassembled) - 1, 0, -1):
            fused_motion = self.motion_fusion_blocks[i](
                fused_motion, reassembled[i - 1]
            )
        fused_motion = self.motion_fusion_blocks[0](fused_motion)

        motion = self.motion_head(fused_motion)  # (B, 6, H, W)

        # Crop back to original size (for intermediate outputs)
        if pad_h > 0 or pad_w > 0:
            motion = motion[..., :H, :W]

        # --- Refinement U-Net ---
        # Takes original input x (unpadded), and the predicted depth/motion
        refined_rgb = self.refinement_net(x, depth, motion)

        # Outputs
        outputs = {
            "depth": depth,
            "motion": motion,
            "sharp_image": refined_rgb,
        }

        if self.cfg.used_image_blurring_block == "pred":
            outputs["blur_image"] = motion_blur(
                refined_rgb,
                depth,
                motion,
                camera_matrix=camera_matrix,
                exposure_time=1.0,
            )
        elif self.cfg.used_image_blurring_block == "GT":
            if gt_sharp is not None:
                outputs["blur_image"] = motion_blur(
                    gt_sharp,
                    depth,
                    motion,
                    camera_matrix=camera_matrix,
                    exposure_time=1.0,
                )

        return outputs

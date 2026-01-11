import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import List, Dict
from omegaconf import DictConfig


class ResidualConvUnit(nn.Module):
    """Residual convolutional unit from RefineNet."""

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
        use_bn: bool = False,
        activation: nn.Module = nn.ReLU(inplace=True),
        deconv: bool = False,
        expand: bool = False,
    ):
        super().__init__()

        out_features = features
        if expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, bias=True)

        self.resConfUnit1 = ResidualConvUnit(features, use_bn, activation)
        self.resConfUnit2 = ResidualConvUnit(features, use_bn, activation)

        if deconv:
            self.upsample = nn.ConvTranspose2d(
                out_features, out_features, kernel_size=2, stride=2
            )
        else:
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        output = xs[0]

        if len(xs) == 2:
            # Resize second input to match first
            res = nn.functional.interpolate(
                xs[1],
                size=(xs[0].shape[2], xs[0].shape[3]),
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

        # Project to output channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Resample to target resolution
        if scale_factor > 1.0:
            self.resample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=int(scale_factor),
                stride=int(scale_factor),
            )
        elif scale_factor < 1.0:
            self.resample = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=int(1.0 / scale_factor),
                padding=1,
            )
        else:
            self.resample = nn.Identity()

    def forward(self, tokens: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        Args:
            tokens: (B, N+1, C) where N is number of patches, +1 for CLS token
            patch_size: Size of each patch in the original image
        """
        B, N_plus_1, C = tokens.shape

        # Separate CLS token and patch tokens
        cls_token = tokens[:, 0:1, :]  # (B, 1, C)
        patch_tokens = tokens[:, 1:, :]  # (B, N, C)

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

        # Reshape to spatial grid
        N = tokens_out.shape[1]
        H = W = int(N**0.5)
        assert H * W == N, f"Number of tokens {N} is not a perfect square"

        tokens_2d = tokens_out.transpose(1, 2).reshape(B, C, H, W)

        # Project and resample
        out = self.proj(tokens_2d)
        out = self.resample(out)

        return out


class DPTHead(nn.Module):
    """Base head for DPT predictions."""

    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class DepthHead(DPTHead):
    """Depth prediction head outputting HxWx1."""

    def __init__(self, in_channels: int, use_bn: bool = False):
        super().__init__(in_channels, out_channels=1, use_bn=use_bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth = super().forward(x)
        # Ensure positive depth values
        depth = F.relu(depth)
        return depth


class MotionHead(DPTHead):
    """Motion prediction head outputting HxWx6 (linear and angular velocity)."""

    def __init__(self, in_channels: int, use_bn: bool = False):
        super().__init__(in_channels, out_channels=6, use_bn=use_bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        motion = super().forward(x)
        return motion


class SharpImageHead(DPTHead):
    """Sharp image prediction head outputting HxWx3."""

    def __init__(self, in_channels: int, use_bn: bool = False):
        super().__init__(in_channels, out_channels=3, use_bn=use_bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sharp = super().forward(x)
        # Ensure valid RGB range [0, 1]
        sharp = torch.sigmoid(sharp)
        return sharp


def se3_exp_map(
    v: torch.Tensor, omega: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SE(3) exponential map: exp(t * ξ̂) where ξ = (v, ω)

    Args:
        v: Linear velocity (B, 3, H, W) or (B, N, H, W, 3)
        omega: Angular velocity (B, 3, H, W) or (B, N, H, W, 3)
        t: Time scalar, (N,) tensor, or (B, N, H, W, 1)

    Returns:
        R: Rotation matrix (B, [N,] H, W, 3, 3)
        T: Translation vector (B, [N,] H, W, 3)
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
            # Will need to add time dimension
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
    """
    Apply motion blur using vectorized SE(3) integration.

    Optimized version that processes all time samples in parallel.
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
    # R: (B, N, H, W, 3, 3), T: (B, N, H, W, 3)

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


class DPT(nn.Module):
    """
    Dense Prediction Transformer with configurable backbone and multiple heads.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Load backbone (frozen or trainable based on config)
        self.backbone = AutoModel.from_pretrained(
            cfg.backbone.name, trust_remote_code=True
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

        # Fusion blocks
        self.fusion_blocks = nn.ModuleList()
        num_layers = len(self.feature_layers)

        for i in range(num_layers):
            self.fusion_blocks.append(
                FeatureFusionBlock(
                    features=cfg.decoder.features,
                    use_bn=cfg.decoder.use_bn,
                    deconv=cfg.decoder.use_deconv,
                    expand=(i != num_layers - 1),
                )
            )

        # Prediction heads
        final_features = cfg.decoder.features // (2 ** (num_layers - 1))

        self.depth_head = DepthHead(in_channels=final_features, use_bn=cfg.heads.use_bn)

        self.motion_head = MotionHead(
            in_channels=final_features, use_bn=cfg.heads.use_bn
        )

        self.sharp_head = SharpImageHead(
            in_channels=final_features, use_bn=cfg.heads.use_bn
        )

        # Motion field solver flag
        self.use_motion_solver = cfg.use_motion_solver

    def forward_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from backbone at multiple layers."""
        outputs = self.backbone(x, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states

        # Extract features from specified layers
        features = []
        for layer_idx in self.feature_layers:
            # layer_idx can be negative (from end) or positive
            features.append(hidden_states[layer_idx])

        return features

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Dictionary containing predictions and intermediate outputs
        """
        # Extract multi-scale features from backbone
        backbone_features = self.forward_backbone(x)

        # Reassemble features into spatial representations
        reassembled = []
        for feat, reassemble_block in zip(backbone_features, self.reassemble_blocks):
            reassembled.append(reassemble_block(feat, self.cfg.backbone.patch_size))

        # Progressive fusion (from deep to shallow)
        fused = reassembled[-1]
        for i in range(len(reassembled) - 1, 0, -1):
            fused = self.fusion_blocks[i](fused, reassembled[i - 1])

        # Final fusion
        fused = self.fusion_blocks[0](fused)

        # Generate predictions from heads
        depth = self.depth_head(fused)  # (B, 1, H, W)
        motion = self.motion_head(fused)  # (B, 6, H, W)
        sharp = self.sharp_head(fused)  # (B, 3, H, W)

        outputs = {
            "depth": depth,
            "motion": motion,
            "sharp_image": sharp,
            "features": fused,
        }

        # Apply motion field solver if configured (non-learnable operation)
        if self.use_blurring_block:
            blurred = motion_blur(depth, motion)
            outputs["blur_image"] = blurred

        return outputs


def build_dpt(cfg: DictConfig) -> DPT:
    """Build DPT model from config."""
    return DPT(cfg)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


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
        target_size: tuple[int, int] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, T, C)
            patch_size: ViT patch size
            image_size: Tuple of (H, W) of the original image
            target_size: Explicit resolution to upsample to
        """
        B, T, C = tokens.shape

        if image_size is not None:
            H_img, W_img = image_size
            H = H_img // patch_size
            W = W_img // patch_size
        else:
            N_spatial = T - 1
            H = W = int(N_spatial**0.5)

        N_spatial = H * W
        cls_token = tokens[:, 0:1, :]
        patch_tokens = tokens[:, -N_spatial:, :]

        if self.read_operation == "ignore":
            tokens_out = patch_tokens
        elif self.read_operation == "add":
            tokens_out = patch_tokens + cls_token
        elif self.read_operation == "project":
            cls_expanded = cls_token.expand(-1, patch_tokens.shape[1], -1)
            tokens_cat = torch.cat([patch_tokens, cls_expanded], dim=-1)
            tokens_out = self.read_proj(tokens_cat)
        else:
            raise ValueError(f"Unknown read operation: {self.read_operation}")

        tokens_2d = tokens_out.transpose(1, 2).reshape(B, tokens_out.shape[-1], H, W)
        out = self.proj(tokens_2d)

        # Resample to exact target size if provided
        if target_size is not None:
            out = F.interpolate(
                out, size=target_size, mode="bilinear", align_corners=True
            )
        elif self.scale_factor != 1.0:
            out = F.interpolate(
                out, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
            )

        return out


class DPTEmbedding(nn.Module):
    def __init__(
        self,
        in_c=3,
        embed_dim=48,
        bias=False,
        backbone_name="facebook/dinov2-small",
        patch_size=14,
    ):
        super(DPTEmbedding, self).__init__()

        # 1. Backbone
        self.patch_size = patch_size
        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(
            self.backbone_name, trust_remote_code=True
        )
        self.backbone_dim = self.backbone.config.hidden_size

        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Reassemble Layers
        self.feature_layers = [2, 5, 8, 11]
        self.target_scales = [4, 8, 16, 32]
        self.reassemble_blocks = nn.ModuleList()

        for scale in self.target_scales:
            self.reassemble_blocks.append(
                ReassembleBlock(
                    in_channels=self.backbone_dim,
                    out_channels=embed_dim,
                    scale_factor=self.patch_size / scale,
                    read_operation="project",
                )
            )

        # 3. Fusion Paths
        self.translation_fusion_blocks = nn.ModuleList()
        self.rotation_fusion_blocks = nn.ModuleList()
        for i in range(len(self.target_scales)):
            skip_ch = embed_dim if i > 0 else 0
            self.translation_fusion_blocks.append(
                FeatureFusionBlock(embed_dim, skip_ch, False, upsample=(i != 0))
            )
            self.rotation_fusion_blocks.append(
                FeatureFusionBlock(embed_dim, skip_ch, False, upsample=(i != 0))
            )

        self.translation_up = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.rotation_up = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        outputs = self.backbone(x_padded, output_hidden_states=True, return_dict=True)
        features = [outputs.hidden_states[i] for i in self.feature_layers]

        # Calculate exact target resolutions for each level to avoid rounding errors
        # reassembled[0] is scale 4, [1] is scale 8, etc.
        targets = [
            (x_padded.shape[2] // scale, x_padded.shape[3] // scale)
            for scale in self.target_scales
        ]

        reassembled = [
            block(f, self.patch_size, image_size=x_padded.shape[2:], target_size=t)
            for f, block, t in zip(features, self.reassemble_blocks, targets)
        ]

        # Fusion
        t_out = reassembled[-1]
        r_out = reassembled[-1]
        for i in range(len(reassembled) - 1, 0, -1):
            t_out = self.translation_fusion_blocks[i](t_out, reassembled[i - 1])
            r_out = self.rotation_fusion_blocks[i](r_out, reassembled[i - 1])

        # Final upsample to match input x (not padded size)
        t_out = F.interpolate(
            self.translation_up(self.translation_fusion_blocks[0](t_out)),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )
        r_out = F.interpolate(
            self.rotation_up(self.rotation_fusion_blocks[0](r_out)),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )

        out = t_out + r_out
        return out

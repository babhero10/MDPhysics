import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# ============================================================================
# 1. VISION TRANSFORMER BACKBONE COMPONENTS
# ============================================================================


class PatchEmbedding(nn.Module):
    """Splits image into patches and embeds them - works with any image size"""

    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolution that splits image into patches and projects to embed_dim
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape

        # Check if image dimensions are divisible by patch_size
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Image dimensions ({H}, {W}) must be divisible by patch_size ({self.patch_size})"

        x = self.proj(x)  # (B, embed_dim, H/P, W/P)

        # Get spatial dimensions after patching
        h, w = x.shape[2], x.shape[3]

        # Flatten spatial dims: (B, embed_dim, H/P, W/P) -> (B, embed_dim, N)
        x = x.flatten(2)
        # Transpose: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.transpose(1, 2)

        return x, (h, w)  # Return tokens and spatial dimensions


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim=768, num_heads=12, qkv_bias=True, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: Q @ K^T / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """Feed-forward network in transformer block"""

    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.0, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block"""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, qkv_bias, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer backbone for DPT - Dynamic image size support"""

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0,
        feature_layers=None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        # Default feature extraction layers (for ViT-Base: layers 3, 6, 9, 12)
        # Using 0-indexed, so [2, 5, 8, 11]
        self.feature_layers = (
            feature_layers if feature_layers is not None else [2, 5, 8, 11]
        )

        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings will be interpolated for different sizes
        # Initialize for a default size, but will adapt dynamically
        self.pos_embed = None  # Will be created on first forward pass

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, mlp_ratio, qkv_bias, dropout, attn_dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x, h, w):
        """Interpolate position embeddings for arbitrary image sizes"""
        npatch = x.shape[1] - 1  # Exclude CLS token
        N = self.pos_embed.shape[1] - 1

        if npatch == N and h == w:
            return self.pos_embed

        # Separate CLS and patch position embeddings
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]

        # Get the height and width of the original position embedding
        h0 = w0 = int(N**0.5)

        # Interpolate patch embeddings
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(h, w), mode="bicubic", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        # Concatenate CLS and patch embeddings
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x, (h, w) = self.patch_embed(x)  # (B, N, embed_dim)

        # Initialize position embeddings if first time
        if self.pos_embed is None:
            # Create position embeddings for the current input size
            n_patches = h * w
            self.pos_embed = nn.Parameter(
                torch.zeros(1, n_patches + 1, self.embed_dim), requires_grad=True
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.pos_embed = self.pos_embed.to(x.device)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)

        # Add interpolated positional embeddings
        x = x + self.interpolate_pos_encoding(x, h, w)

        # Store intermediate features for DPT
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.feature_layers:
                features.append(
                    (self.norm(x), h, w)
                )  # Store normalized features with spatial dims

        return features


# ============================================================================
# 2. REASSEMBLE BLOCKS - Convert tokens back to feature maps
# ============================================================================


class Reassemble(nn.Module):
    """Reassembles patch tokens into spatial feature maps with optional upsampling
    Now works dynamically with any spatial dimensions"""

    def __init__(self, embed_dim, out_channels, upsample_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.upsample_factor = upsample_factor

        # Projection to desired output channels
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        # Upsampling if needed
        if upsample_factor > 1:
            self.upsample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=upsample_factor,
                stride=upsample_factor,
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, x, h, w):
        """
        x: (B, N+1, embed_dim) - tokens including CLS
        h, w: spatial dimensions of the patch grid
        """
        B = x.shape[0]

        # Remove CLS token
        x = x[:, 1:, :]  # (B, N, embed_dim)

        # Reshape to spatial dimensions
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h, w)

        # Project to output channels
        x = self.proj(x)

        # Upsample if needed
        x = self.upsample(x)

        return x


# ============================================================================
# 3. FUSION BLOCKS - Progressively fuse multi-scale features
# ============================================================================


class ResidualConvUnit(nn.Module):
    """Residual convolutional unit used in fusion blocks"""

    def __init__(self, channels, use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=not use_batch_norm
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=not use_batch_norm
        )
        self.relu = nn.ReLU(inplace=True)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(channels)
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        return x + out  # Residual connection


class FeatureFusionBlock(nn.Module):
    """Fuses features from different scales"""

    def __init__(self, channels, use_batch_norm=False, shared_params=False):
        super().__init__()
        self.shared_params = shared_params

        if shared_params:
            # Share the same ResidualConvUnit for both operations
            self.resConfUnit = ResidualConvUnit(channels, use_batch_norm)
        else:
            # Separate ResidualConvUnits
            self.resConfUnit1 = ResidualConvUnit(channels, use_batch_norm)
            self.resConfUnit2 = ResidualConvUnit(channels, use_batch_norm)

    def forward(self, x, residual=None):
        # Apply first residual conv unit
        if self.shared_params:
            output = self.resConfUnit(x)
        else:
            output = self.resConfUnit1(x)

        # Add residual from higher-resolution features if provided
        if residual is not None:
            # Upsample if sizes don't match
            if output.shape[-2:] != residual.shape[-2:]:
                residual = F.interpolate(
                    residual,
                    size=output.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            output = output + residual

        # Apply second residual conv unit
        if self.shared_params:
            output = self.resConfUnit(output)
        else:
            output = self.resConfUnit2(output)

        return output


# ============================================================================
# 4. DPT HEAD - Final prediction layer
# ============================================================================


class DPTHead(nn.Module):
    """Prediction head for DPT (e.g., depth estimation, segmentation)"""

    def __init__(self, in_channels, out_channels=1, num_conv_layers=3):
        super().__init__()

        layers = []
        current_channels = in_channels

        # Build convolutional layers
        for i in range(num_conv_layers - 1):
            next_channels = current_channels // 2 if i == 0 else 32
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels, next_channels, kernel_size=3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = next_channels

        # Final prediction layer
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


# ============================================================================
# 5. COMPLETE DPT MODEL - Dynamic & Flexible
# ============================================================================


class DPT(nn.Module):
    """
    Dense Prediction Transformer - Complete Model

    Features:
    - Works with any image size (divisible by patch_size)
    - Configurable number of heads
    - Option for shared/separate fusion parameters
    - Flexible feature extraction layers
    - Multiple upsampling strategies
    """

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        decoder_channels=256,
        out_channels=1,
        feature_layers=None,
        upsample_factors=None,
        shared_fusion_params=False,
        use_batch_norm=False,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        """
        Args:
            patch_size: Size of image patches (16 or 8 typical)
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Transformer embedding dimension (768 for Base, 1024 for Large)
            depth: Number of transformer blocks (12 for Base, 24 for Large)
            num_heads: Number of attention heads (12 for Base, 16 for Large)
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            decoder_channels: Number of channels in decoder
            out_channels: Number of output channels (1 for depth, N for segmentation)
            feature_layers: Which transformer layers to extract features from
            upsample_factors: Upsampling factor for each feature level
            shared_fusion_params: Whether to share parameters in fusion blocks
            use_batch_norm: Whether to use batch normalization in fusion blocks
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout rate in MLP
            attn_dropout: Dropout rate in attention
        """
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels

        # Default feature extraction layers (0-indexed)
        if feature_layers is None:
            # For ViT-Base (depth=12): extract from layers 3, 6, 9, 12 -> [2, 5, 8, 11]
            # Adjust based on depth
            step = max(1, depth // 4)
            feature_layers = [i * step + (step - 1) for i in range(4)]
            feature_layers = [min(l, depth - 1) for l in feature_layers]

        self.feature_layers = feature_layers
        self.num_feature_levels = len(feature_layers)

        # Default upsampling factors (creates multi-scale pyramid)
        if upsample_factors is None:
            upsample_factors = [4, 2, 1, 1]  # Standard DPT configuration

        assert (
            len(upsample_factors) == self.num_feature_levels
        ), f"upsample_factors length ({len(upsample_factors)}) must match feature_layers ({self.num_feature_levels})"

        # 1. Vision Transformer Backbone
        self.backbone = VisionTransformer(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attn_dropout=attn_dropout,
            feature_layers=feature_layers,
        )

        # 2. Reassemble blocks for each feature level
        self.reassemble_blocks = nn.ModuleList(
            [
                Reassemble(embed_dim, decoder_channels, upsample_factor)
                for upsample_factor in upsample_factors
            ]
        )

        # 3. Fusion blocks - progressively fuse features
        self.fusion_blocks = nn.ModuleList(
            [
                FeatureFusionBlock(
                    decoder_channels, use_batch_norm, shared_fusion_params
                )
                for _ in range(self.num_feature_levels)
            ]
        )

        # 4. Final prediction head
        self.head = DPTHead(decoder_channels, out_channels)

    def forward(self, x):
        """
        Forward pass - works with any image size

        Args:
            x: Input tensor (B, C, H, W) where H and W are divisible by patch_size

        Returns:
            output: Prediction tensor (B, out_channels, H, W)
        """
        input_size = x.shape[-2:]

        # 1. Extract multi-scale features from ViT backbone
        features = self.backbone(x)  # List of (feature, h, w) tuples

        # 2. Reassemble tokens to feature maps
        reassembled = []
        for i, (feat, h, w) in enumerate(features):
            reassembled.append(self.reassemble_blocks[i](feat, h, w))

        # 3. Progressively fuse features (bottom-up)
        # Start from the deepest (smallest) feature and work up
        out = self.fusion_blocks[-1](reassembled[-1])

        for i in range(self.num_feature_levels - 2, -1, -1):
            out = self.fusion_blocks[i](reassembled[i], out)

        # 4. Generate prediction
        out = self.head(out)

        # 5. Upsample to input resolution
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return out

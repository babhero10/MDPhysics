import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
import math

from .archs.dpt_arch import DPTEmbedding
from .archs.mdt_arch import TransformerBlock, Downsample, Upsample, PatchEmbed, Fuse

from omegaconf import ListConfig


class mdt(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12, 8],
        num_refinement_blocks=4,
        ffn_expansion_factor=3,
        bias=False,
        img_size=128,
        backbone_name="facebook/dinov2-small",
        patch_size=14,
    ):
        super(mdt, self).__init__()

        distortion_model = "polynomial"
        norm_layer = nn.LayerNorm

        # 1. Determine Padded Image Size at Init
        # We must align the architecture to the padded size, not the raw input size.
        # Both H and W must be divisible by patch_size * 4 (for 2 levels of 2x downsampling).
        stride = patch_size * 4

        if isinstance(img_size, (list, tuple, ListConfig)):
            h, w = int(img_size[0]), int(img_size[1])
        else:
            h, w = int(img_size), int(img_size)

        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride

        self.padded_H = h + pad_h
        self.padded_W = w + pad_w

        # 2. Calculate Cuts based on PADDED size
        radius_cuts = self.padded_H // patch_size
        azimuth_cuts = self.padded_W // patch_size
        res = max(radius_cuts, azimuth_cuts)

        n_radius = 1
        n_azimuth = 1
        cartesian = (
            torch.cartesian_prod(torch.linspace(-1, 1, res), torch.linspace(1, -1, res))
            .reshape(res, res, 2)
            .transpose(2, 1)
            .transpose(1, 0)
            .transpose(1, 2)
        )
        radius = cartesian.norm(dim=0)
        theta = torch.atan2(cartesian[1], cartesian[0])

        self.patch_embed0 = DPTEmbedding(
            inp_channels,
            dim,
            bias=bias,
            backbone_name=backbone_name,
            patch_size=patch_size,
        )

        self.patch_embed = PatchEmbed(
            img_size=(self.padded_H, self.padded_W),  # Use padded size
            distortion_model=distortion_model,
            radius_cuts=radius_cuts,
            azimuth_cuts=azimuth_cuts,
            radius=radius,
            azimuth=theta,
            in_chans=inp_channels,
            embed_dim=dim,
            n_radius=n_radius,
            n_azimuth=n_azimuth,
            norm_layer=norm_layer,
        )

        self.ape = False
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=0.0)

        # Store configuration for _make_layer
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.bias = bias
        self.patch_size_polar = self.patch_embed.patch_size

        # Drop path rates
        drop_path_rate = 0.1
        self.dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))
        ]

        patches_resolution = self.patch_embed.patches_resolution

        # Resolutions for each level (Symmetric Downsampling)
        res_level1 = (patches_resolution[0], patches_resolution[1])
        res_level2 = (patches_resolution[0] // 2, patches_resolution[1] // 2)
        res_level3 = (patches_resolution[0] // 4, patches_resolution[1] // 4)

        # --- Encoder Levels ---
        self.encoder_level1 = self._make_layer(
            dim=dim,
            num_blocks=num_blocks[0],
            resolution=res_level1,
            dpr_slice=self.dpr[: num_blocks[0]],
            att=False,
        )

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = self._make_layer(
            dim=int(dim * 2),
            num_blocks=num_blocks[1],
            resolution=res_level2,
            dpr_slice=self.dpr[num_blocks[0] : num_blocks[0] + num_blocks[1]],
            att=False,
        )

        self.down2_3 = Downsample(int(dim * 2))

        self.encoder_level3 = self._make_layer(
            dim=int(dim * 4),
            num_blocks=num_blocks[2],
            resolution=res_level3,
            dpr_slice=self.dpr[
                num_blocks[0]
                + num_blocks[1] : num_blocks[0]
                + num_blocks[1]
                + num_blocks[2]
            ],
            att=False,
        )

        # --- Decoder Levels ---
        self.decoder_level3 = self._make_layer(
            dim=int(dim * 4),
            num_blocks=num_blocks[2],
            resolution=res_level3,
            dpr_slice=self.dpr[
                num_blocks[0]
                + num_blocks[1] : num_blocks[0]
                + num_blocks[1]
                + num_blocks[2]
            ],
            att=True,
        )

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 4), int(dim * 2), kernel_size=1, bias=bias
        )

        self.decoder_level2 = self._make_layer(
            dim=int(dim * 2),
            num_blocks=num_blocks[1],
            resolution=res_level2,
            dpr_slice=self.dpr[num_blocks[0] : num_blocks[0] + num_blocks[1]],
            att=True,
        )

        self.up2_1 = Upsample(int(dim * 2))

        self.decoder_level1 = self._make_layer(
            dim=dim,
            num_blocks=num_blocks[0],
            resolution=res_level1,
            dpr_slice=self.dpr[: num_blocks[0]],
            att=True,
        )

        self.refinement = self._make_layer(
            dim=dim,
            num_blocks=num_refinement_blocks,
            resolution=res_level1,
            dpr_slice=self.dpr[
                :num_refinement_blocks
            ],  # Uses same slice as Enc1 as per original code
            att=True,
        )

        self.fuse2 = Fuse(dim * 2, res_level2, self.patch_size_polar)
        self.fuse1 = Fuse(dim, res_level1, self.patch_size_polar)
        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def _make_layer(self, dim, num_blocks, resolution, dpr_slice, att):
        layers = []
        for i in range(num_blocks):
            layers.append(
                TransformerBlock(
                    input_resolution=resolution,
                    dim=dim,
                    ffn_expansion_factor=self.ffn_expansion_factor,
                    bias=self.bias,
                    att=att,
                    drop_path=dpr_slice[i] if i < len(dpr_slice) else 0.0,
                    patch_size=self.patch_size_polar,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, inp_img, dist=None):
        if dist is None:
            B = inp_img.shape[0]
            dist = torch.zeros((B, 4), device=inp_img.device)

        # Pad Input to match the architecture initialized in __init__
        B, C, H, W = inp_img.shape

        # Calculate padding to reach the initialized padded size
        # Note: This assumes input H, W are <= initialized sizes.
        # If input is larger (e.g. testing), we should dynamically pad to nearest multiple
        # but the window sizes in TransformerBlock are fixed at init.
        # This is a limitation of this specific WindowAttention implementation (it bakes in pos embeddings).
        # For now, we pad to the size calculated at init (optimized for 360x640).

        # However, to be robust to varying test sizes, we should probably check if H/W match config.
        # But let's stick to the training fix first.

        pad_h = self.padded_H - H
        pad_w = self.padded_W - W

        # If padding is negative (input larger than init), this might fail.
        # But we assume config matches data.
        if pad_h < 0 or pad_w < 0:
            # Fallback for larger images: pad to nearest stride, but this will crash WindowAttention
            # because it expects fixed window size.
            # The only way to support variable resolution is if WindowAttention supported it.
            # Current WindowAttention has learnable params of fixed size (a_p, b_p).
            # So we MUST pad/crop to exactly self.padded_H/W or integer multiples of windows?
            # Actually, WindowAttention uses 'window_size' which is fixed.
            # If the feature map is larger, window_partition just makes MORE windows.
            # The problem is 'window_size' itself.
            # Level 3 window width = (padded_W // patch_size) // 4.
            # This is a single huge window covering the whole width.
            # If input width changes, this window size is wrong.
            # So, for THIS specific architecture (Stripe Attention), input size is effectively fixed/locked at init.
            pass

        if pad_h > 0 or pad_w > 0:
            inp_img = F.pad(inp_img, (0, pad_w, 0, pad_h), mode="reflect")

        inp_enc_level1, D_s, theta_max = self.patch_embed(inp_img, dist)
        inp_enc_level1 = self.patch_embed0(inp_img)

        if self.ape:
            inp_enc_level1 = inp_enc_level1 + self.absolute_pos_embed

        inp_enc_level1 = self.pos_drop(inp_enc_level1)

        inp_enc1 = [inp_enc_level1, D_s, theta_max]
        out_enc_level1 = self.encoder_level1(inp_enc1)

        inp_enc_level2 = self.down1_2(out_enc_level1[0])
        inp_enc2 = [inp_enc_level2, out_enc_level1[1], theta_max]
        out_enc_level2 = self.encoder_level2(inp_enc2)

        inp_enc_level3 = self.down2_3(out_enc_level2[0])
        inp_enc3 = [inp_enc_level3, out_enc_level2[1], theta_max]
        out_enc_level3 = self.encoder_level3(inp_enc3)

        inp_dec3 = [out_enc_level3[0], out_enc_level3[1], theta_max]
        out_dec_level3 = self.decoder_level3(inp_dec3)

        inp_dec_level2 = self.up3_2(out_dec_level3[0])
        inp_dec_level2 = [inp_dec_level2, out_dec_level3[1], theta_max]
        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        inp_dec2 = [inp_dec_level2[0], out_dec_level3[1], theta_max]
        out_dec_level2 = self.decoder_level2(inp_dec2)

        inp_dec_level1 = self.up2_1(out_dec_level2[0])
        inp_dec_level1 = [inp_dec_level1, out_dec_level2[1], theta_max]
        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)

        inp_dec1 = [inp_dec_level1[0], out_dec_level2[1], theta_max]
        out_dec_level1 = self.decoder_level1(inp_dec1)

        inp_dec1 = [out_dec_level1[0], out_dec_level1[1], theta_max]
        out_dec_level1 = self.refinement(inp_dec1)

        out_dec_level1 = self.output(out_dec_level1[0]) + inp_img

        # Crop Output
        if pad_h > 0 or pad_w > 0:
            out_dec_level1 = out_dec_level1[..., :H, :W]

        return [out_dec_level1]


class DPT(mdt):
    """Wrapper to align MDT with the config structure."""

    def __init__(self, cfg):
        dim = 48
        num_blocks = [6, 6, 12, 8]
        num_refinement_blocks = 4
        ffn_expansion_factor = 3
        backbone_name = "facebook/dinov2-small"
        patch_size = 14
        img_size = 128

        if hasattr(cfg, "dim"):
            dim = cfg.dim
        if hasattr(cfg, "num_blocks"):
            num_blocks = cfg.num_blocks
        if hasattr(cfg, "num_refinement_blocks"):
            num_refinement_blocks = cfg.num_refinement_blocks
        if hasattr(cfg, "ffn_expansion_factor"):
            ffn_expansion_factor = cfg.ffn_expansion_factor
        if hasattr(cfg, "backbone"):
            if hasattr(cfg.backbone, "name"):
                backbone_name = cfg.backbone.name
            if hasattr(cfg.backbone, "patch_size"):
                patch_size = cfg.backbone.patch_size
        if hasattr(cfg, "img_size"):
            # Handle list [H, W] or int, including ListConfig
            val = cfg.img_size
            img_size = val

        super().__init__(
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            backbone_name=backbone_name,
            patch_size=patch_size,
            img_size=img_size,
        )
        self.cfg = cfg

    def forward(self, x, dist=None):
        if dist is None:
            B = x.shape[0]
            dist = torch.zeros((B, 4), device=x.device)

        out_list = super().forward(x, dist)
        out = out_list[0]

        return {"sharp_image": out}

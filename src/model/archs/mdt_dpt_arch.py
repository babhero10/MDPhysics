import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForDepthEstimation
from einops import rearrange
import math
from timm.layers import trunc_normal_

# Import existing MDT components
from .mdt_arch import (
    TransformerBlock,
    Downsample,
    Upsample,
    Fuse,
    PatchEmbed,
    get_sample_params_from_subdiv
)

class ReassembleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, h, w, target_size):
        # x: (B, N, C)
        # target_size: (H_out, W_out)
        if x.dim() == 3:
            B, N, C = x.shape
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        x = self.project(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class FusionPathCartesian(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        out = x + skip
        out = self.res_conv(out)
        out = self.act(out)
        return out

class FusionPathPolar(nn.Module):
    def __init__(self, channels, patch_size, input_resolution):
        super().__init__()
        # MDT TransformerBlock (Polar) initialization is based on fixed resolutions
        # to determine window_size.
        # This effectively sets the "stripe width" for attention.
        self.block = TransformerBlock(
            patch_size=patch_size,
            input_resolution=input_resolution,
            dim=channels,
            ffn_expansion_factor=2.66,
            bias=False,
            att=True 
        )
        
    def forward(self, x, skip, polar_params):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
        fused = x + skip
        out = self.block([fused, polar_params[0], polar_params[1]])
        return out[0]

class FusionBlock(nn.Module):
    def __init__(self, channels, patch_size, resolution):
        super().__init__()
        self.cartesian = FusionPathCartesian(channels)
        self.polar = FusionPathPolar(channels, patch_size, resolution)
    
    def forward(self, x_t, x_r, skip, polar_params):
        out_t = self.cartesian(x_t, skip)
        out_r = self.polar(x_r, skip, polar_params)
        return out_t, out_r

class MotionDPT(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        
        self.dim = dim 
        
        backbone_name = "facebook/dinov2-base"
        depth_name = "depth-anything/Depth-Anything-V2-Small-hf"
        self.out_indices = [8, 9, 10, 11]
        
        if hasattr(cfg, 'backbone_name'): backbone_name = cfg.backbone_name
        if hasattr(cfg, 'depth_backbone_name'): depth_name = cfg.depth_backbone_name
        if hasattr(cfg, 'out_indices'): self.out_indices = cfg.out_indices
        
        # Load Backbones
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_name)
        
        # Freeze Backbones
        freeze_backbones = True
        if hasattr(cfg, 'freeze_backbones'): freeze_backbones = cfg.freeze_backbones
        elif isinstance(cfg, dict) and 'freeze_backbones' in cfg: freeze_backbones = cfg['freeze_backbones']
            
        if freeze_backbones:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.depth_model.parameters():
                param.requires_grad = False
        
        embed_dim = self.backbone.config.hidden_size
        
        self.reassemble = nn.ModuleList([
            ReassembleBlock(embed_dim, self.dim),
            ReassembleBlock(embed_dim, self.dim),
            ReassembleBlock(embed_dim, self.dim),
            ReassembleBlock(embed_dim, self.dim),
        ])
        
        # Base resolution used ONLY for defining the Window Size structure in Polar blocks
        base_res = 128
        if hasattr(cfg, 'patch_size'): base_res = cfg.patch_size
        
        # Structure resolutions (for initializing Window sizes)
        # These define the minimum attention window size.
        struct_resolutions = [
            (base_res // 32, base_res // 32),
            (base_res // 16, base_res // 16),
            (base_res // 8, base_res // 8),
            (base_res // 4, base_res // 4),
        ]
        
        polar_patch_size = [base_res, base_res]
        
        self.fusion_blocks = nn.ModuleList()
        self.fusion_blocks.append(FusionBlock(self.dim, polar_patch_size, struct_resolutions[1]))
        self.fusion_blocks.append(FusionBlock(self.dim, polar_patch_size, struct_resolutions[2]))
        self.fusion_blocks.append(FusionBlock(self.dim, polar_patch_size, struct_resolutions[3]))
        
        self.head_upsample = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(self.dim, self.dim, 3, 1, 1)
        )
        
        self.rsas = TransformerBlock(
            patch_size=polar_patch_size,
            input_resolution=(base_res, base_res),
            dim=self.dim,
            att=True
        )
        
    def get_polar_params(self, x_shape, device):
        B, C, H, W = x_shape
        subdiv = (H, W) 
        dist = torch.zeros((B, 4), device=device).transpose(1, 0)
        params, D_s, theta_max = get_sample_params_from_subdiv(
            subdiv=subdiv,
            img_size=(H, W),
            distortion_model="polynomial",
            D=dist,
            n_radius=1, 
            n_azimuth=1,
            radius_buffer=0,
            azimuth_buffer=0
        )
        return D_s, theta_max

    def forward(self, x):
        # x is (B, 3, H, W)
        # MDT_DPT_Impl ensures H is % 32, W is % 128.
        # But DINOv2 needs % 14.
        
        B, C, H, W = x.shape
        
        # 1. Pad for DINOv2 (Multiple of 14)
        h_pad = (14 - H % 14) % 14
        w_pad = (14 - W % 14) % 14
        
        x_dino = x
        if h_pad > 0 or w_pad > 0:
            x_dino = F.pad(x, (0, w_pad, 0, h_pad), mode="reflect")
        
        H_dino, W_dino = x_dino.shape[-2:]
        
        # 2. Run DINOv2
        outputs = self.backbone(x_dino, output_hidden_states=True)
        states = outputs.hidden_states
        selected_states = [states[i] for i in self.out_indices]
        
        h_feat = H_dino // 14
        w_feat = W_dino // 14
        
        # 3. Define Dynamic Target Resolutions based on ORIGINAL input x
        # This preserves the 1280x720 scale through the network.
        # Scales: 1/32, 1/16, 1/8, 1/4 of the input H, W
        target_resolutions = [
            (H // 32, W // 32),
            (H // 16, W // 16),
            (H // 8, W // 8),
            (H // 4, W // 4)
        ]
        
        # 4. Reassemble
        feats = []
        for i, block in enumerate(self.reassemble):
            # Slice features to match h_feat * w_feat (removes CLS etc)
            features = selected_states[i][:, -h_feat*w_feat:, :]
            # block interpolates to target_resolutions[i]
            # This implicitly handles the unpadding of the DINOv2 extra pixels
            f = block(features, h=h_feat, w=w_feat, target_size=target_resolutions[i])
            feats.append(f)
            
        x_t = feats[0]
        x_r = feats[0] 
        
        for i, block in enumerate(self.fusion_blocks):
            skip = feats[i+1]
            D_s, theta_max = self.get_polar_params(skip.shape, x.device)
            polar_params = [D_s, theta_max]
            x_t, x_r = block(x_t, x_r, skip, polar_params)
            
        x_t = self.head_upsample(x_t)
        x_r = self.head_upsample(x_r)
        
        # 5. Depth Integration
        # Depth model is robust to size, but let's be safe and use x (padded for MDT)
        depth_out = self.depth_model(x).predicted_depth
        depth = depth_out.unsqueeze(1)
        
        # Align depth to x_t (which is H, W)
        if depth.shape[-2:] != x_t.shape[-2:]:
             depth = F.interpolate(depth, size=x_t.shape[-2:], mode='bilinear', align_corners=False)
        
        depth = depth + 1e-5
        x_t = x_t / depth
        
        x_sum = x_t + x_r
        
        D_s_full, theta_max_full = self.get_polar_params(x_sum.shape, x.device)
        x_sum = self.rsas([x_sum, D_s_full, theta_max_full])[0]
        
        return x_sum

class MDT_DPT_Impl(nn.Module):
    def __init__(
        self,
        cfg,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 6, 12, 8],
        num_refinement_blocks=4,
        ffn_expansion_factor=3,
        bias=False,
        patch_size=128,
    ):
        super(MDT_DPT_Impl, self).__init__()

        res = patch_size
        distortion_model = "polynomial"
        norm_layer = nn.LayerNorm
        radius_cuts = res
        azimuth_cuts = res
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

        self.patch_embed0 = MotionDPT(dim, cfg)

        self.patch_embed = PatchEmbed(
            img_size=patch_size,
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

        drop_rate = 0.0
        self.ape = False
        num_patches = self.patch_embed.num_patches
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        drop_path_rate = 0.1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        patches_resolution = self.patch_embed.patches_resolution
        patch_size_val = self.patch_embed.patch_size

        patches_res_num = 4
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**0),
                        patches_resolution[1] // (patches_res_num**0),
                    ),
                    dim=dim,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    drop_path=dpr[sum(num_blocks[:0]) : sum(num_blocks[: 0 + 1])][i],
                    patch_size=patch_size_val,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**1),
                        patches_resolution[1] // (patches_res_num**1),
                    ),
                    dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    drop_path=dpr[sum(num_blocks[:1]) : sum(num_blocks[: 0 + 2])][i],
                    patch_size=patch_size_val,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**2),
                        patches_resolution[1] // (patches_res_num**2),
                    ),
                    dim=int(dim * 2**2),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    drop_path=dpr[sum(num_blocks[:2]) : sum(num_blocks[: 0 + 3])][i],
                    patch_size=patch_size_val,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**2),
                        patches_resolution[1] // (patches_res_num**2),
                    ),
                    dim=int(dim * 2**2),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                    drop_path=dpr[sum(num_blocks[:2]) : sum(num_blocks[: 0 + 3])][i],
                    patch_size=patch_size_val,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**1),
                        patches_resolution[1] // (patches_res_num**1),
                    ),
                    dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                    drop_path=dpr[sum(num_blocks[:1]) : sum(num_blocks[: 0 + 2])][i],
                    patch_size=patch_size_val,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**0),
                        patches_resolution[1] // (patches_res_num**0),
                    ),
                    dim=int(dim),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                    drop_path=dpr[sum(num_blocks[:0]) : sum(num_blocks[: 0 + 1])][i],
                    patch_size=patch_size_val,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**0),
                        patches_resolution[1] // (patches_res_num**0),
                    ),
                    dim=int(dim),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                    drop_path=dpr[sum(num_blocks[:0]) : sum(num_blocks[: 0 + 1])][i],
                    patch_size=patch_size_val,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.fuse2 = Fuse(dim * 2, 2, patches_resolution, patch_size_val)
        self.fuse1 = Fuse(dim, 1, patches_resolution, patch_size_val)
        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img, dist=None):
        if dist is None:
            dist = torch.zeros((inp_img.shape[0], 4), device=inp_img.device)

        pad_h_mult = int(self.patch_embed.radius_cuts)
        pad_w_mult = int(self.patch_embed.azimuth_cuts)

        _, _, H, W = inp_img.shape
        pad_h = (pad_h_mult - H % pad_h_mult) % pad_h_mult
        pad_w = (pad_w_mult - W % pad_w_mult) % pad_w_mult

        H_orig, W_orig = H, W

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

        if pad_h > 0 or pad_w > 0:
            out_dec_level1 = out_dec_level1[:, :, :H_orig, :W_orig]

        return [out_dec_level1]
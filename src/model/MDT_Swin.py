import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torchvision.ops import DeformConv2d

from timm.layers import DropPath, to_2tuple, trunc_normal_
from .utils.polar_utils import get_sample_params_from_subdiv, get_sample_locations
import numpy as np

pi = 3.141592653589793


class DConv(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.conv2 = DeformConv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def theta(window_size, num_heads, radius, theta_max, a_r, b_r, H, total_batch_size):
    if theta_max.numel() == 0:
        return torch.zeros(
            (
                total_batch_size,
                num_heads,
                window_size[0] * window_size[1],
                window_size[0] * window_size[1],
            ),
            device=radius.device,
        )

    a_r = a_r[radius]
    b_r = b_r[radius]

    B = theta_max.shape[0]
    if B == 0:
        return torch.zeros(
            (
                total_batch_size,
                num_heads,
                window_size[0] * window_size[1],
                window_size[0] * window_size[1],
            ),
            device=radius.device,
        )

    nW = total_batch_size // B

    theta_max = theta_max.repeat_interleave(nW, dim=0).reshape(
        total_batch_size, 1, 1, 1
    )

    radius_float = radius.float().unsqueeze(0).unsqueeze(-1)

    radius_new = radius_float * theta_max / H

    A_r = a_r.unsqueeze(0) * torch.cos(radius_new) + b_r.unsqueeze(0) * torch.sin(
        radius_new
    )

    return A_r.permute(0, 3, 1, 2)


def phi(window_size, num_heads, azimuth, a_p, b_p, W):
    a_p = a_p[azimuth]
    b_p = b_p[azimuth]
    azimuth = azimuth * 2 * np.pi / W
    azimuth = azimuth[:, :, None].repeat(1, 1, num_heads)

    A_phi = a_p * torch.cos(azimuth) + b_p * torch.sin(azimuth)
    return A_phi


def window_partition(x, window_size):
    B, H, W, C = x.shape
    if type(window_size) is tuple:
        x = x.view(
            B,
            H // window_size[0],
            window_size[0],
            W // window_size[1],
            window_size[1],
            C,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0], window_size[1], C)
        )
        return windows
    else:
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size, window_size, C)
        )
        return windows


def window_reverse(windows, window_size, H, W):
    if type(window_size) is tuple:
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        x = windows.view(
            B,
            H // window_size[0],
            W // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
    else:
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(
            B, H // window_size, W // window_size, window_size, window_size, -1
        )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        patch_size,
        input_resolution,
        dim,
        window_size,
        num_heads=8,  # IMPROVED: Multi-head attention (default 8 instead of 1)
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_heads = num_heads  # IMPROVED: Configurable multi-head
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        H, W = input_resolution

        if input_resolution == window_size:
            self.a_p = nn.Parameter(torch.zeros(window_size[1], self.num_heads))
            self.b_p = nn.Parameter(torch.zeros(window_size[1], self.num_heads))
        else:
            self.a_p = nn.Parameter(
                torch.zeros((2 * window_size[1] - 1), self.num_heads)
            )
            self.b_p = nn.Parameter(
                torch.zeros((2 * window_size[1] - 1), self.num_heads)
            )
        self.a_r = nn.Parameter(torch.zeros((2 * window_size[0] - 1), self.num_heads))
        self.b_r = nn.Parameter(torch.zeros((2 * window_size[0] - 1), self.num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        radius = relative_coords[:, :, 0]
        azimuth = relative_coords[:, :, 1]
        r_max = self.patch_size[0] * H
        self.r_max = r_max
        self.register_buffer("radius", radius)
        self.register_buffer("azimuth", azimuth)

        self.qkv = nn.Linear(dim, dim * 3, bias=4)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.a_p, std=0.02)
        trunc_normal_(self.a_r, std=0.02)
        trunc_normal_(self.b_p, std=0.02)
        trunc_normal_(self.b_r, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, theta_max, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        A_phi = phi(
            self.window_size,
            self.num_heads,
            self.azimuth,
            self.a_p,
            self.b_p,
            self.input_resolution[1],
        )
        A_theta = theta(
            self.window_size,
            self.num_heads,
            self.radius,
            theta_max,
            self.a_r,
            self.b_r,
            self.input_resolution[0],
            B_,
        )
        attn = attn + A_phi.transpose(1, 2).transpose(0, 1).unsqueeze(0) + A_theta

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.fft = nn.Parameter(
            torch.ones(
                (hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)
            )
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(
            x,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(
            x_patch,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(
            dim * 6,
            dim * 6,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 6,
            bias=bias,
        )

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type="WithBias")

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(
            q,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        k_patch = rearrange(
            k,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(
            out,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(
        self,
        patch_size,
        input_resolution,
        dim,
        num_heads=8,  # IMPROVED: Multi-head parameter
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        att=False,
        drop_path=0.0,
    ):
        super(TransformerBlock, self).__init__()
        self.window_size = (1, input_resolution[1])
        self.input_resolution = input_resolution
        self.att = att
        if self.att:
            self.norm1 = nn.LayerNorm(dim)
            self.patch_size = patch_size
            self.attn = WindowAttention(
                self.patch_size,
                self.input_resolution,
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,  # IMPROVED: Pass multi-head parameter
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
            )

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)
        self.attn_mask = None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, inp):
        x = inp[0]
        D_s = inp[1]
        theta_max = inp[2]

        if self.att:
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)

            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            shifted_x = x

            x_windows = window_partition(shifted_x, self.window_size)
            if type(self.window_size) is tuple:
                x_windows = x_windows.view(
                    -1, self.window_size[0] * self.window_size[1], C
                )
                attn_windows = self.attn(x_windows, theta_max, mask=self.attn_mask)
                attn_windows = attn_windows.view(
                    -1, self.window_size[0], self.window_size[1], C
                )
            else:
                x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
                attn_windows = self.attn(x_windows, mask=self.attn_mask)
                attn_windows = attn_windows.view(
                    -1, self.window_size, self.window_size, C
                )

            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
            x = x.view(B, H * W, C)

            x = shortcut + self.drop_path(x)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.ffn(self.norm2(x))
        out = [x, D_s, theta_max]
        return out


class Fuse(nn.Module):
    def __init__(self, n_feat, l, patches_resolution, patch_size, num_heads=8):
        super(Fuse, self).__init__()

        self.n_feat = n_feat
        self.att_channel = TransformerBlock(
            patch_size=patch_size,
            input_resolution=(
                patches_resolution[0] // (1**l),
                patches_resolution[1] // (4**l),
            ),
            dim=n_feat * 2,
            num_heads=num_heads,  # IMPROVED: Pass num_heads
        )

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        _ = self.conv(torch.cat((enc[0], dnc[0]), dim=1))
        x = [_, enc[1], enc[2]]
        x = self.att_channel(x)
        _ = self.conv2(x[0])
        e, d = torch.split(_, [self.n_feat, self.n_feat], dim=1)
        output = e + d
        x = [output, enc[1], enc[2]]
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.dconv = DeformConv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.dcon_c = DConv(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.polar_offset_conv = nn.Conv2d(
            in_c, 2 * 3 * 3, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        out2 = self.conv_polar_fixed(x, self.polar_offset_conv)
        out1 = self.dcon_c(x)
        out2 = self.dconv(x, out2)
        out = out1 + out2
        return out

    def conv_polar_fixed(self, input_tensor, conv_layer):
        res = conv_layer(input_tensor)
        return F.softmax(res, dim=1)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.body(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.body(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=128,
        distortion_model="polynomial",
        radius_cuts=16,
        azimuth_cuts=64,
        n_radius=8,
        n_azimuth=16,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.distortion_model = distortion_model
        self.subdiv = (radius_cuts, azimuth_cuts)
        self.n_radius = n_radius
        self.n_azimuth = n_azimuth

        patch_size = [self.img_size[0] / (2 * radius_cuts), 2 * pi / azimuth_cuts]
        self.patch_size = patch_size

        self.azimuth_cuts = azimuth_cuts
        self.radius_cuts = radius_cuts

        patches_resolution = [
            radius_cuts,
            azimuth_cuts,
        ]  ### azimuth is always cut in even partition
        self.patches_resolution = patches_resolution
        self.num_patches = radius_cuts * azimuth_cuts

    def forward(self, x, dist):
        # We only need H and W to define the coordinate space
        _, _, H, W = x.shape

        # Transpose dist and get parameters
        # params is ignored as you only want D_s and theta_max
        _, D_s, theta_max = get_sample_params_from_subdiv(
            subdiv=self.subdiv,
            img_size=(H, W),
            distortion_model=self.distortion_model,
            D=dist.transpose(1, 0),
            n_radius=self.n_radius,
            n_azimuth=self.n_azimuth,
            radius_buffer=0,
            azimuth_buffer=0,
        )

        return D_s, theta_max


class mdt(nn.Module):
    def __init__(self, cfg):
        super(mdt, self).__init__()

        inp_channels = cfg.get("inp_channels", 3)
        out_channels = cfg.get("out_channels", 3)
        dim = cfg.get("dim", 48)
        num_blocks = cfg.get("num_blocks", [6, 6, 12, 8])
        num_refinement_blocks = cfg.get("num_refinement_blocks", 4)
        ffn_expansion_factor = cfg.get("ffn_expansion_factor", 3)
        bias = cfg.get("bias", False)
        img_size = cfg.get("img_size", 128)
        num_heads = cfg.get("num_heads", 8)  # IMPROVED: Multi-head attention

        res = img_size
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

        self.patch_embed0 = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            distortion_model=distortion_model,
            radius_cuts=radius_cuts,
            azimuth_cuts=azimuth_cuts,
            n_radius=n_radius,
            n_azimuth=n_azimuth,
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
                    num_heads=num_heads,  # IMPROVED: Multi-head attention
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
                    num_heads=num_heads,  # IMPROVED: Multi-head attention
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
                    num_heads=num_heads,  # IMPROVED: Multi-head attention
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
                    num_heads=num_heads,  # IMPROVED: Multi-head attention
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    att=True,
                    drop_path=dpr[sum(num_blocks[:0]) : sum(num_blocks[: 0 + 1])][i],
                    patch_size=patch_size_val,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.fuse2 = Fuse(
            dim * 2, 2, patches_resolution, patch_size_val, num_heads
        )  # IMPROVED
        self.fuse1 = Fuse(
            dim, 1, patches_resolution, patch_size_val, num_heads
        )  # IMPROVED
        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img, dist=None):
        if dist is None:
            dist = torch.zeros((inp_img.shape[0], 4), device=inp_img.device)

        # Determine padding multiples from patch_size configuration
        # self.patch_embed.patch_size is [H_multiple, W_multiple]?
        # In init: patch_size = [self.img_size[0]/(2*radius_cuts), self.max_azimuth/azimuth_cuts]
        # Wait, the MDT logic for patch_size in init is complicated/polar specific.

        # Let's look at how Encoder Level 1 is initialized:
        # patches_resolution = self.patch_embed.patches_resolution (radius_cuts, azimuth_cuts)
        # We know Level 3 downsamples by 4. DFFN uses patch_size=8.
        # So we need divisibility by 4 * 8 = 32 for Height.

        # For Width (Azimuth), the stripe window size is critical.
        # In Encoder Level 1: window_size is (1, input_resolution[1]).
        # input_resolution is patches_resolution[1] (azimuth_cuts).
        # We need the Width to be divisible by the effective stride?
        # Actually, if window_size is (1, W), we just need W to match?
        # The crash was in window_partition.

        # Let's rely on the patch_size passed to __init__ which is used to set 'res'.
        # We need to ensure we pad to multiples that satisfy the UNet + Window Attention structure.

        # Based on previous context:
        # Height needs % 32.
        # Width needs % 128 (likely related to window_size being 128?).
        # If we want this DYNAMIC based on config 'patch_size', we need to check how 'patch_size' affects model structure.

        # In MDT_Edited: patch_size is passed to mdt init.
        # In mdt.__init__:
        # res = patch_size
        # radius_cuts = res
        # azimuth_cuts = res
        # ...
        # self.patch_embed = PatchEmbed(img_size=patch_size, ..., radius_cuts=res, azimuth_cuts=res, ...)

        # In PatchEmbed:
        # patches_resolution = [radius_cuts, azimuth_cuts] = [res, res]

        # In mdt.__init__ again:
        # patches_res_num = 4
        # self.encoder_level1: input_resolution = (res // 1, res // 1)
        # self.encoder_level1 -> TransformerBlock -> WindowAttention(..., window_size=(1, input_resolution[1]))
        # So window_size is (1, res).

        # So yes, the Width must be divisible by 'res' (which is the configured patch_size, e.g., 128).
        # And Height?
        # Level 3 is res // 4.
        # DFFN in Level 3 uses patch_size=8.
        # So (res // 4) must be divisible by 8?
        # If res=128, 128//4 = 32. 32%8 == 0. OK.
        # But this is about the *model structure*.
        # The INPUT IMAGE size divisibility requirement is what we are padding for.

        # If the model expects windows of size 'res' (e.g. 128) in Width, then Input Width must be divisible by 128?
        # Actually, window_partition splits (B, H, W, C) into windows.
        # If window_size is (1, W_win), then W must be divisible by W_win.
        # Here W_win = res = 128.
        # So Input Width must be % 128.

        # For Height:
        # Level 3 downsamples by 4.
        # DFFN rearranges with patch1=8.
        # So (H / 4) must be divisible by 8. => H must be divisible by 32.

        # So we should use:
        # pad_h_mult = 32
        # pad_w_mult = self.patch_embed.radius_cuts # which is 'res' i.e. configured patch_size

        # Wait, 'radius_cuts' is used for Height in PatchEmbed logic?
        # patches_resolution = [radius_cuts, azimuth_cuts].
        # In mdt: input_resolution=(patches_resolution[0], patches_resolution[1]).
        # (H, W).
        # So window_size[1] comes from azimuth_cuts = res.

        pad_h_mult = 32
        pad_w_mult = int(
            self.patch_embed.azimuth_cuts
        )  # This is the configured 'patch_size' (128)

        _, _, H, W = inp_img.shape
        pad_h = (pad_h_mult - H % pad_h_mult) % pad_h_mult
        pad_w = (pad_w_mult - W % pad_w_mult) % pad_w_mult

        H_orig, W_orig = H, W

        if pad_h > 0 or pad_w > 0:
            inp_img = F.pad(inp_img, (0, pad_w, 0, pad_h), mode="reflect")

        D_s, theta_max = self.patch_embed(inp_img, dist)
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


# Wrapper class for compatibility with training pipeline
class MDT_Swin(mdt):
    """MDT with Multi-Head Attention improvements (simplified version)

    Key improvements over original MDT:
    1. Multi-head attention (8 heads instead of 1) for better feature representation
    2. Uses full-width windows (1, W) - same as original MDT
    3. Stable and works with any resolution at inference

    Args:
        cfg: Configuration dictionary with keys:
            - img_size: Image size (default: 128)
            - num_heads: Number of attention heads (default: 8)
            - dim: Base dimension (default: 48)
            - num_blocks: Blocks per level (default: [6, 6, 12, 8])
            - Other standard MDT parameters
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        out_list = super().forward(x, dist=None)
        return {"sharp_image": out_list[0]}

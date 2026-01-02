import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torchvision.ops import DeformConv2d

from timm.layers import DropPath, to_2tuple, trunc_normal_
from .polar_utils import get_sample_params_from_subdiv, get_sample_locations
from .physics_mdm import PhysicsInformedMDM
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


def R(window_size, num_heads, radius, D, a_r, b_r, r_max):
    a_r = a_r[radius.view(-1)].reshape(
        window_size[0] * window_size[1], window_size[0] * window_size[1], num_heads
    )
    b_r = b_r[radius.view(-1)].reshape(
        window_size[0] * window_size[1], window_size[0] * window_size[1], num_heads
    )
    radius = radius[None, :, None, :].repeat(num_heads, 1, D.shape[0], 1)
    radius = D * radius

    radius = radius.transpose(0, 1).transpose(1, 2).transpose(2, 3).transpose(0, 1)

    A_r = a_r * torch.cos(radius * 2 * pi / r_max) + b_r * torch.sin(
        radius * 2 * pi / r_max
    )

    return A_r


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
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    if isinstance(window_size, tuple):
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
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """

    if isinstance(window_size, tuple):
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
    r"""Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(
        self,
        patch_size,
        input_resolution,
        dim,
        window_size,
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
        self.num_heads = 1
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

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2

        radius = relative_coords[:, :, 0]
        azimuth = relative_coords[:, :, 1]

        r_max = self.patch_size[0] * H
        self.r_max = r_max

        self.register_buffer("radius", radius, persistent=False)
        self.register_buffer("azimuth", azimuth, persistent=False)

        # FIXED: bias=True instead of bias=4
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.a_p, std=0.02)
        trunc_normal_(self.a_r, std=0.02)
        trunc_normal_(self.b_p, std=0.02)
        trunc_normal_(self.b_r, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, theta_max, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        # Ensure num_heads compatibility. If num_heads=1, shapes are simple.
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
        # FIXED: Correct LayerNorm behavior by subtracting mean
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight


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
        # Ensure compatibility with complex numbers
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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        patch_size,
        input_resolution,
        dim,
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
        # We now expect inp to be a list [x, D_s, theta_max] because we will handle list passing manually in ModuleList loops
        x = inp[0]  # BCHW
        D_s = inp[1]
        theta_max = inp[2]

        if self.att:
            x = x.permute(0, 2, 3, 1)  # BHWC
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)

            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            shifted_x = x

            x_windows = window_partition(shifted_x, self.window_size)
            if isinstance(self.window_size, tuple):
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
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # BCHW

        x = x + self.ffn(self.norm2(x))
        out = [x, D_s, theta_max]
        return out


class Fuse(nn.Module):
    def __init__(self, n_feat, l, patches_resolution, patch_size):
        super(Fuse, self).__init__()

        self.n_feat = n_feat
        self.att_channel = TransformerBlock(
            patch_size=patch_size,
            input_resolution=(
                patches_resolution[0] // (1**l),
                patches_resolution[1] // (4**l),
            ),
            dim=n_feat * 2,
        )

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        # enc and dnc are lists [x, D_s, theta_max]
        _ = self.conv(torch.cat((enc[0], dnc[0]), dim=1))
        x = [_, enc[1], enc[2]]
        x = self.att_channel(x)
        _ = self.conv2(x[0])
        e, d = torch.split(_, [self.n_feat, self.n_feat], dim=1)
        output = e + d
        x = [output, enc[1], enc[2]]
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False, fan_regions=1):
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

        # FIXED: Initialize conv layer here to ensure weights are learned.
        self.fan_regions = fan_regions
        self.polar_conv = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        kernel_size = 3
        # Use internal method for polar conv to access self.polar_conv
        out2 = self._conv_polar(x, self.fan_regions, 2 * kernel_size * kernel_size)

        out1 = self.dcon_c(x)
        out2 = self.dconv(x, out2)
        out = out1 + out2
        return out

    def _conv_polar(self, input_tensor, fan_regions, embed_dim):
        b, c, h, w = input_tensor.size()
        start_angle = 0
        end_angle = 2 * np.pi / fan_regions
        center_x = w // 2
        center_y = h // 2
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=input_tensor.device),
            torch.arange(0, w, device=input_tensor.device),
            indexing="ij",
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()
        dx = grid_x - center_x
        dy = grid_y - center_y
        angle = torch.atan2(dy, dx)
        fan_idx = torch.floor(
            (angle - start_angle + np.pi) / (end_angle - start_angle)
        ).long()
        fan_conv_results = torch.zeros(
            b, embed_dim, h, w, dtype=torch.float, device=input_tensor.device
        )
        softmax = nn.Softmax(dim=1)

        for i in range(fan_regions):
            if i == fan_regions - 1:
                mask1 = fan_idx == i
                mask2 = fan_idx == i + 1
                mask = (mask1 + mask2).to(input_tensor.device)
            else:
                mask = (fan_idx == i).to(input_tensor.device)
            fan_input = input_tensor * mask
            # Use the learned layer
            fan_conv_result = self.polar_conv(fan_input) * mask
            fan_conv_results += fan_conv_result

        fan_conv_results = softmax(fan_conv_results)
        return fan_conv_results


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
        radius=None,
        azimuth=None,
        in_chans=3,
        embed_dim=96,
        n_radius=8,
        n_azimuth=16,
        norm_layer=None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)

        patches_resolution = [radius_cuts, azimuth_cuts]
        self.azimuth_cuts = azimuth_cuts
        self.radius_cuts = radius_cuts
        self.subdiv = (self.radius_cuts, self.azimuth_cuts)
        self.img_size = img_size
        self.distoriton_model = distortion_model
        self.radius = radius
        self.azimuth = azimuth
        self.max_azimuth = np.pi * 2
        patch_size = [
            self.img_size[0] / (2 * radius_cuts),
            self.max_azimuth / azimuth_cuts,
        ]

        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = radius_cuts * azimuth_cuts
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.n_radius = n_radius
        self.n_azimuth = n_azimuth
        self.mlp = nn.Linear(self.n_radius * self.n_azimuth * in_chans, embed_dim)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, dist):
        B, C, H, W = x.shape

        dist = dist.transpose(1, 0)
        radius_buffer, azimuth_buffer = 0, 0
        params, D_s, theta_max = get_sample_params_from_subdiv(
            subdiv=self.subdiv,
            img_size=self.img_size,
            distortion_model=self.distoriton_model,
            D=dist,
            n_radius=self.n_radius,
            n_azimuth=self.n_azimuth,
            radius_buffer=radius_buffer,
            azimuth_buffer=azimuth_buffer,
        )

        sample_locations = get_sample_locations(**params, img_B=B)
        B2, n_p, n_s = sample_locations[0].shape

        x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
        x_ = x_ / (H // 2)
        y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()
        y_ = y_ / (W // 2)
        out = torch.cat((y_, x_), dim=3)
        out = out.to(x.device)

        x_out = torch.empty(
            B, self.embed_dim, self.radius_cuts, self.azimuth_cuts, device=x.device
        )

        tensor = (
            F.grid_sample(x, out, align_corners=True)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, self.n_radius * self.n_azimuth * self.in_chans)
        )

        out_ = self.mlp(tensor)
        out_ = out_.contiguous().view(B, self.radius_cuts * self.azimuth_cuts, -1)

        out_up = out_.reshape(B, self.azimuth_cuts, self.radius_cuts, self.embed_dim)
        out_up = out_up.transpose(1, 3)

        x_out[:, :, : self.radius_cuts, :] = out_up
        x = x_out.permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x, D_s, theta_max

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class MDT(nn.Module):
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
    ):
        super(MDT, self).__init__()

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
        self.patch_norm = True

        self.patch_embed0 = PhysicsInformedMDM(
            inp_channels, dim, img_size, img_size, focal_length=100.0
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
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
        patch_size = self.patch_embed.patch_size
        patches_res_num = 4

        # Use ModuleList instead of Sequential for custom looping with multiple arguments
        self.encoder_level1 = nn.ModuleList(
            [
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**0),
                        patches_resolution[1] // (patches_res_num**0),
                    ),
                    dim=dim,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    drop_path=dpr[sum(num_blocks[:0]) : sum(num_blocks[: 0 + 1])][i],
                    patch_size=patch_size,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.ModuleList(
            [
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**1),
                        patches_resolution[1] // (patches_res_num**1),
                    ),
                    dim=int(dim * 2**1),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    drop_path=dpr[sum(num_blocks[:1]) : sum(num_blocks[: 0 + 2])][i],
                    patch_size=patch_size,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.ModuleList(
            [
                TransformerBlock(
                    input_resolution=(
                        patches_resolution[0] // (1**2),
                        patches_resolution[1] // (patches_res_num**2),
                    ),
                    dim=int(dim * 2**2),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    drop_path=dpr[sum(num_blocks[:2]) : sum(num_blocks[: 0 + 3])][i],
                    patch_size=patch_size,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.decoder_level3 = nn.ModuleList(
            [
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
                    patch_size=patch_size,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.ModuleList(
            [
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
                    patch_size=patch_size,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.ModuleList(
            [
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
                    patch_size=patch_size,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.ModuleList(
            [
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
                    patch_size=patch_size,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.fuse2 = Fuse(dim * 2, 2, patches_resolution, patch_size)
        self.fuse1 = Fuse(dim, 1, patches_resolution, patch_size)
        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward_features(self, module_list, x_pack):
        # Helper to iterate over ModuleList with packed arguments
        for block in module_list:
            x_pack = block(x_pack)
        return x_pack

    def forward(self, inp_img, dist):
        # 1. Polar Embeddings
        inp_enc_polar, D_s, theta_max = self.patch_embed(inp_img, dist)

        # 2. Physics Embeddings (Fixes overwrite bug)
        inp_enc_physics, inv_Z, velocity = self.patch_embed0(inp_img)

        # 3. Fuse Embeddings (Add them)
        inp_enc_level1 = inp_enc_polar + inp_enc_physics

        if self.ape:
            inp_enc_level1 = inp_enc_level1 + self.absolute_pos_embed

        inp_enc_level1 = self.pos_drop(inp_enc_level1)

        # Packed input for Transformers
        inp_enc1 = [inp_enc_level1, D_s, theta_max]

        # Encoder 1
        out_enc_level1 = self.forward_features(self.encoder_level1, inp_enc1)

        # Encoder 2
        inp_enc_level2 = self.down1_2(out_enc_level1[0])
        inp_enc2 = [inp_enc_level2, out_enc_level1[1], theta_max]
        out_enc_level2 = self.forward_features(self.encoder_level2, inp_enc2)

        # Encoder 3
        inp_enc_level3 = self.down2_3(out_enc_level2[0])
        inp_enc3 = [inp_enc_level3, out_enc_level2[1], theta_max]
        out_enc_level3 = self.forward_features(self.encoder_level3, inp_enc3)

        # Decoder 3
        inp_dec3 = [out_enc_level3[0], out_enc_level3[1], theta_max]
        out_dec_level3 = self.forward_features(self.decoder_level3, inp_dec3)

        # Decoder 2
        inp_dec_level2 = self.up3_2(out_dec_level3[0])
        inp_dec_level2 = [inp_dec_level2, out_dec_level3[1], theta_max]
        inp_dec_level2 = self.fuse2(
            inp_dec_level2, out_enc_level2
        )  # fuse handles lists internally

        inp_dec2 = [inp_dec_level2[0], out_dec_level3[1], theta_max]
        out_dec_level2 = self.forward_features(self.decoder_level2, inp_dec2)

        # Decoder 1
        inp_dec_level1 = self.up2_1(out_dec_level2[0])
        inp_dec_level1 = [inp_dec_level1, out_dec_level2[1], theta_max]
        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)

        inp_dec1 = [inp_dec_level1[0], out_dec_level2[1], theta_max]
        out_dec_level1 = self.forward_features(self.decoder_level1, inp_dec1)

        # Refinement
        inp_refine = [out_dec_level1[0], out_dec_level1[1], theta_max]
        out_refine = self.forward_features(self.refinement, inp_refine)

        # Output
        final_out = self.output(out_refine[0]) + inp_img

        return [final_out, inv_Z, velocity]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torchvision.ops import DeformConv2d
from timm.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.utils.checkpoint as checkpoint

from ..utils.polar_utils import get_sample_params_from_subdiv, get_sample_locations

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
        self.window_size = window_size  # Wh, Ww
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

        # Dynamic Interpolation of Position Embeddings
        # x is (B*nW, N, C). Window size corresponds to N.
        # We assume square windows or strip windows.
        # Current logic assumes N = Wh * Ww.
        # If window_size is tuple (1, W), then N = W.

        # Determine current window dimensions
        if self.window_size[0] == 1:  # Stripe Attention (Height 1)
            current_h = 1
            current_w = N
        else:
            # Assume square-ish or proportional?
            # If dynamic, we might not know H/W ratio from N alone without more info.
            # However, for this specific MDPhysics architecture, Level 3 is (1, W).
            # Levels 1 & 2 might be square blocks?
            # Let's check init logic.
            # If init was (1, 16), it's Stripe.
            # If init was (8, 8), it's Block.

            # Safe fallback: If N matches init, use init.
            if N == self.window_size[0] * self.window_size[1]:
                current_h, current_w = self.window_size
            else:
                # If mismatch, we assume it scales proportionally or is a strip?
                # Given MDT uses Stripe Attention (H=1) for the bottleneck,
                # we prioritize that logic.
                if self.window_size[0] == 1:
                    current_h = 1
                    current_w = N
                else:
                    # For square windows, window_partition fixes the size.
                    # So N should NOT change for square windows (tiling handles it).
                    # The only case N changes is for Global/Stripe windows.
                    # So we assume the mismatch is due to Width scaling.
                    current_h = self.window_size[0]
                    current_w = N // current_h

        # Interpolate a_p, b_p (Azimuth / Width)
        if current_w != self.window_size[1]:
            # Reshape to (1, C, L) for interpolation
            # a_p is (W, nH) -> (1, nH, W)
            a_p_img = self.a_p.permute(1, 0).unsqueeze(0)
            b_p_img = self.b_p.permute(1, 0).unsqueeze(0)

            a_p = F.interpolate(
                a_p_img, size=current_w, mode="linear", align_corners=False
            )
            b_p = F.interpolate(
                b_p_img, size=current_w, mode="linear", align_corners=False
            )

            # Back to (W, nH)
            a_p = a_p.squeeze(0).permute(1, 0)
            b_p = b_p.squeeze(0).permute(1, 0)

            # Update azimuth tensor for new width
            # We need to regenerate relative coords?
            # Or just scale the existing azimuth logic?
            # 'phi' function uses 'azimuth' index.
            # We need 'azimuth' to be indices 0..current_w-1

            # Generate new azimuth indices dynamically
            # This is expensive to do every forward, but correct.
            coords_w = torch.arange(current_w, device=x.device)
            # If H>1, we need full meshgrid.
            if current_h > 1:
                coords_h = torch.arange(current_h, device=x.device)
                coords = torch.stack(
                    torch.meshgrid([coords_h, coords_w], indexing="ij")
                )
                coords_flatten = torch.flatten(coords, 1)
                relative_coords = (
                    coords_flatten[:, :, None] - coords_flatten[:, None, :]
                )
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()
                current_azimuth = relative_coords[:, :, 1]
                current_radius = relative_coords[:, :, 0]
            else:
                # Optimized for H=1 (Stripe)
                # relative_coords is just w_diff
                coords_flatten = coords_w.unsqueeze(0)  # (1, W)
                current_azimuth = (
                    coords_flatten[:, :, None] - coords_flatten[:, None, :]
                )  # (1, W, 1) - (1, 1, W) -> (1, W, W)
                current_azimuth = current_azimuth.squeeze(0)  # (W, W)
                # Radius is all zeros for H=1 relative diff in H dim?
                # Wait, radius in WindowAttention is 'relative_coords[:, :, 0]' which is H dimension diff.
                # If H=1, H diff is 0.
                current_radius = torch.zeros((N, N), device=x.device, dtype=torch.long)

        else:
            a_p, b_p = self.a_p, self.b_p
            current_azimuth = self.azimuth
            current_radius = self.radius

        # Interpolate a_r, b_r (Radius / Height)
        if current_h != self.window_size[0]:
            # Similar interpolation for H dimension parameters
            # a_r is (H, nH) -> (1, nH, H)
            a_r_img = self.a_r.permute(1, 0).unsqueeze(0)
            b_r_img = self.b_r.permute(1, 0).unsqueeze(0)

            a_r = F.interpolate(
                a_r_img, size=current_h, mode="linear", align_corners=False
            )
            b_r = F.interpolate(
                b_r_img, size=current_h, mode="linear", align_corners=False
            )

            a_r = a_r.squeeze(0).permute(1, 0)
            b_r = b_r.squeeze(0).permute(1, 0)
        else:
            a_r, b_r = self.a_r, self.b_r

        A_phi = phi(
            (current_h, current_w),
            self.num_heads,
            current_azimuth,
            a_p,
            b_p,
            self.input_resolution[
                1
            ],  # Use original normalization? Or dynamic? Original is safer for scale.
        )
        A_theta = theta(
            (current_h, current_w),
            self.num_heads,
            current_radius,
            theta_max,
            a_r,
            b_r,
            self.input_resolution[0],
            B_,
        )
        attn = attn + A_phi.transpose(1, 2).transpose(0, 1).unsqueeze(0) + A_theta

        if mask is not None:
            # Mask shape might need update if N changed?
            # Usually mask is None for Global attention.
            # If mask is present, we assume window size hasn't changed (Tiling case).
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
        use_checkpoint=False,
    ):
        super(TransformerBlock, self).__init__()
        self.use_checkpoint = use_checkpoint
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

    def _forward_impl(self, x, D_s, theta_max):
        # Implementation of the forward pass
        if self.att:
            x_in = x
            x = x.permute(0, 2, 3, 1)  # BHWC
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
            B, L, C = x.shape
            # assert L == H * W, "input feature has wrong size"

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            shifted_x = x

            # Debug print

            # Dynamic window size for Stripe Attention
            current_window_size = self.window_size
            if isinstance(self.window_size, tuple) and self.window_size[0] == 1:
                 # If it's a stripe (H=1), adapt Width to current feature map width
                 if self.window_size[1] != W:
                     current_window_size = (1, W)

            x_windows = window_partition(shifted_x, current_window_size)

            if type(current_window_size) is tuple:
                x_windows = x_windows.view(
                    -1, current_window_size[0] * current_window_size[1], C
                )
                attn_windows = self.attn(x_windows, theta_max, mask=self.attn_mask)
                attn_windows = attn_windows.view(
                    -1, current_window_size[0], current_window_size[1], C
                )
            else:
                x_windows = x_windows.view(-1, current_window_size * current_window_size, C)
                attn_windows = self.attn(x_windows, mask=self.attn_mask)
                attn_windows = attn_windows.view(
                    -1, current_window_size, current_window_size, C
                )

            shifted_x = window_reverse(attn_windows, current_window_size, H, W)
            x = shifted_x
            x = x.view(B, H * W, C)

            x = shortcut + self.drop_path(x)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, inp):
        x = inp[0]  # BCHW
        D_s = inp[1]
        theta_max = inp[2]

        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, D_s, theta_max, use_reentrant=False)
        else:
            x = self._forward_impl(x, D_s, theta_max)
            
        out = [x, D_s, theta_max]
        return out


class Fuse(nn.Module):
    def __init__(self, n_feat, resolution, patch_size):
        super(Fuse, self).__init__()

        self.n_feat = n_feat
        self.att_channel = TransformerBlock(
            patch_size=patch_size,
            input_resolution=resolution,
            dim=n_feat * 2,
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
    r"""Image to Patch Embedding"""

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
        device = x.device

        # Dynamically calculate cuts based on input size and configured stride
        # We assume the patch_size[0] (H_pixel / H_cut) is constant.
        # self.patch_size stores [H_pixel_per_cut, W_radian_per_cut].
        # Wait, self.patch_size[0] is float.
        # Better: We know from init that H / radius_cuts = pixels_per_cut.
        # Let's derive effective stride.

        # Stride per cut (in pixels)
        stride_h = self.img_size[0] / self.radius_cuts
        stride_w = self.img_size[1] / self.azimuth_cuts

        # Calculate new cuts
        current_radius_cuts = int(H / stride_h)
        current_azimuth_cuts = int(W / stride_w)

        current_subdiv = (current_radius_cuts, current_azimuth_cuts)

        dist = dist.transpose(1, 0)
        radius_buffer, azimuth_buffer = 0, 0
        params, D_s, theta_max = get_sample_params_from_subdiv(
            subdiv=current_subdiv,
            img_size=(H, W),  # Use current image size
            distortion_model=self.distoriton_model,
            D=dist,
            n_radius=self.n_radius,
            n_azimuth=self.n_azimuth,
            radius_buffer=radius_buffer,
            azimuth_buffer=azimuth_buffer,
        )

        # Optimization: The feature extraction part of PatchEmbed (Polar Grid Sample + MLP)
        # is unused in MDPhysics because it is overwritten by DPTEmbedding.
        # We skip the expensive sampling and return x directly to save compute/memory.
        # The critical outputs here are D_s and theta_max for WindowAttention.
        
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

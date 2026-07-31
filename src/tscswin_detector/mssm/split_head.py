from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Drop paths per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AxisStripeAttention3D(nn.Module):
    """
    Stripe attention along one 3D axis.

    Inputs and outputs use channel-last volume layout: (B, D, H, W, C_axis).
    When q/k/v are supplied, the first positional argument is interpreted as q.
    """

    valid_axes = {"D", "H", "W"}

    def __init__(
        self,
        dim: int,
        D_size: int,
        H_size: int,
        W_size: int,
        num_heads: int,
        axis: str,
        stripe_window_D: int = 2,
        stripe_window_H: int = 2,
        stripe_window_W: int = 2,
        attn_drop: float = 0.0,
        shift_enabled: bool = False,
        shift_size_D: int = 0,
        shift_size_H: int = 0,
        shift_size_W: int = 0,
    ):
        super().__init__()
        if axis not in self.valid_axes:
            raise ValueError(f"axis must be one of {sorted(self.valid_axes)}, got {axis!r}.")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.D_size = D_size
        self.H_size = H_size
        self.W_size = W_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.axis = axis
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.shift_enabled = bool(shift_enabled)

        if axis == "D":
            kernel_size = (3, 1, 1)
            padding = (1, 0, 0)
            self.stripe_length = D_size
            self.plane_size = (H_size, W_size)
            self.plane_window = (stripe_window_H, stripe_window_W)
            self.shift_size = (0, int(shift_size_H), int(shift_size_W))
        elif axis == "H":
            kernel_size = (1, 3, 1)
            padding = (0, 1, 0)
            self.stripe_length = H_size
            self.plane_size = (D_size, W_size)
            self.plane_window = (stripe_window_D, stripe_window_W)
            self.shift_size = (int(shift_size_D), 0, int(shift_size_W))
        else:
            kernel_size = (1, 1, 3)
            padding = (0, 0, 1)
            self.stripe_length = W_size
            self.plane_size = (D_size, H_size)
            self.plane_window = (stripe_window_D, stripe_window_H)
            self.shift_size = (int(shift_size_D), int(shift_size_H), 0)

        if self.plane_window[0] <= 0 or self.plane_window[1] <= 0:
            raise ValueError("stripe window sizes must be positive.")
        if self.plane_size[0] % self.plane_window[0] != 0:
            raise ValueError(
                f"{axis}-axis plane size {self.plane_size[0]} must be divisible by window {self.plane_window[0]}."
            )
        if self.plane_size[1] % self.plane_window[1] != 0:
            raise ValueError(
                f"{axis}-axis plane size {self.plane_size[1]} must be divisible by window {self.plane_window[1]}."
            )
        if any(shift < 0 for shift in self.shift_size):
            raise ValueError(f"shift sizes must be non-negative, got {self.shift_size}.")
        for shift, size, window in zip(
            self.shift_size,
            (D_size, H_size, W_size),
            (stripe_window_D, stripe_window_H, stripe_window_W),
        ):
            if shift > 0 and shift >= window:
                raise ValueError(f"shift size {shift} must be smaller than its window size {window}.")
            if shift > 0 and size <= window:
                raise ValueError(f"shift size {shift} requires spatial size {size} to be larger than window {window}.")
        self.shift_enabled = self.shift_enabled and any(shift > 0 for shift in self.shift_size)

        self.lepe_conv = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:
        q = x
        if k is None:
            k = x
        if v is None:
            v = x

        B, D, H, W, C = q.shape
        if (D, H, W) != (self.D_size, self.H_size, self.W_size):
            raise ValueError(
                f"Expected spatial size {(self.D_size, self.H_size, self.W_size)}, got {(D, H, W)}."
            )
        if C != self.dim:
            raise ValueError(f"Expected channel dim {self.dim}, got {C}.")

        lepe = self._compute_lepe(v)
        if self.shift_enabled:
            q = self._roll_volume(q, reverse=False)
            k = self._roll_volume(k, reverse=False)
            v = self._roll_volume(v, reverse=False)
            lepe = self._roll_volume(lepe, reverse=False)

        q = q.reshape(B, D, H, W, self.num_heads, self.head_dim)
        k = k.reshape(B, D, H, W, self.num_heads, self.head_dim)
        v = v.reshape(B, D, H, W, self.num_heads, self.head_dim)
        lepe = lepe.reshape(B, D, H, W, self.num_heads, self.head_dim)

        q = self._volume_to_axis(q)
        k = self._volume_to_axis(k)
        v = self._volume_to_axis(v)
        lepe = self._volume_to_axis(lepe)

        q_windows = self._partition_axis_windows(q)
        k_windows = self._partition_axis_windows(k)
        v_windows = self._partition_axis_windows(v)
        lepe_windows = self._partition_axis_windows(lepe)

        q_windows = q_windows * self.scale
        attn = q_windows @ k_windows.transpose(-2, -1)
        if self.shift_enabled:
            attn = self._apply_shift_mask(attn, B)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out_windows = (attn @ v_windows) + lepe_windows
        out = self._merge_axis_windows(out_windows, B)
        out = self._axis_to_volume(out)
        out = out.reshape(B, D, H, W, C)
        if self.shift_enabled:
            out = self._roll_volume(out, reverse=True)
        return out

    def _compute_lepe(self, v: torch.Tensor) -> torch.Tensor:
        v = v.permute(0, 4, 1, 2, 3).contiguous()
        lepe = self.lepe_conv(v)
        return lepe.permute(0, 2, 3, 4, 1).contiguous()

    def _roll_volume(self, x: torch.Tensor, reverse: bool) -> torch.Tensor:
        shifts = self.shift_size if reverse else tuple(-shift for shift in self.shift_size)
        return torch.roll(x, shifts=shifts, dims=(1, 2, 3))

    def _volume_to_axis(self, x: torch.Tensor) -> torch.Tensor:
        if self.axis == "D":
            return x
        if self.axis == "H":
            return x.permute(0, 2, 1, 3, 4, 5).contiguous()
        return x.permute(0, 3, 1, 2, 4, 5).contiguous()

    def _axis_to_volume(self, x: torch.Tensor) -> torch.Tensor:
        if self.axis == "D":
            return x
        if self.axis == "H":
            return x.permute(0, 2, 1, 3, 4, 5).contiguous()
        return x.permute(0, 2, 3, 1, 4, 5).contiguous()

    def _partition_axis_windows(self, x: torch.Tensor) -> torch.Tensor:
        B, stripe_len, plane_1, plane_2, num_heads, head_dim = x.shape
        window_1, window_2 = self.plane_window
        num_win_1 = plane_1 // window_1
        num_win_2 = plane_2 // window_2
        tokens = stripe_len * window_1 * window_2

        x = x.reshape(B, stripe_len, num_win_1, window_1, num_win_2, window_2, num_heads, head_dim)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.reshape(B * num_win_1 * num_win_2, num_heads, tokens, head_dim)

    def _merge_axis_windows(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        window_1, window_2 = self.plane_window
        plane_1, plane_2 = self.plane_size
        num_win_1 = plane_1 // window_1
        num_win_2 = plane_2 // window_2

        x = x.reshape(
            batch_size,
            num_win_1,
            num_win_2,
            self.num_heads,
            self.stripe_length,
            window_1,
            window_2,
            self.head_dim,
        )
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(batch_size, self.stripe_length, plane_1, plane_2, self.num_heads, self.head_dim)

    def _apply_shift_mask(self, attn: torch.Tensor, batch_size: int) -> torch.Tensor:
        mask = self._get_shift_mask(attn.device)
        if mask is None:
            return attn

        num_windows = mask.shape[0]
        tokens = mask.shape[-1]
        attn = attn.reshape(batch_size, num_windows, self.num_heads, tokens, tokens)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(2), torch.finfo(attn.dtype).min)
        return attn.reshape(batch_size * num_windows, self.num_heads, tokens, tokens)

    def _get_shift_mask(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.shift_enabled:
            return None

        stripe_len = self.stripe_length
        plane_1, plane_2 = self.plane_size
        shift_1, shift_2 = self._axis_plane_shift()
        window_1, window_2 = self.plane_window

        mask = torch.zeros((1, stripe_len, plane_1, plane_2, 1, 1), device=device)
        slices_1 = self._mask_slices(plane_1, window_1, shift_1)
        slices_2 = self._mask_slices(plane_2, window_2, shift_2)
        count = 0
        for slice_1 in slices_1:
            for slice_2 in slices_2:
                mask[:, :, slice_1, slice_2, :, :] = count
                count += 1

        mask_windows = self._partition_axis_windows(mask).reshape(-1, stripe_len * window_1 * window_2)
        return mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)

    def _axis_plane_shift(self) -> Tuple[int, int]:
        if self.axis == "D":
            return self.shift_size[1], self.shift_size[2]
        if self.axis == "H":
            return self.shift_size[0], self.shift_size[2]
        return self.shift_size[0], self.shift_size[1]

    @staticmethod
    def _mask_slices(size: int, window: int, shift: int):
        if shift <= 0:
            return (slice(0, size),)
        return (slice(0, -window), slice(-window, -shift), slice(-shift, None))


class SplitHeadCSwinBlock3D(nn.Module):
    """Split-head 3D CSWin block with D/H/W head groups."""

    def __init__(
        self,
        dim: int,
        D_size: int,
        H_size: int,
        W_size: int,
        num_heads: int = 8,
        axis_heads: Tuple[int, int, int] = (2, 3, 3),
        stripe_window_D: int = 2,
        stripe_window_H: int = 2,
        stripe_window_W: int = 2,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        gamma_init: float = 0.0,
        shift_enabled: bool = False,
        shift_size_D: int = 0,
        shift_size_H: int = 0,
        shift_size_W: int = 0,
        return_axis_desc: bool = False,
        axis_desc_mode: str = "var_range",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        if len(axis_heads) != 3:
            raise ValueError("axis_heads must contain three values for D, H and W.")
        if any(int(heads) <= 0 for heads in axis_heads):
            raise ValueError(f"axis_heads values must be positive, got {axis_heads}.")
        if sum(axis_heads) != num_heads:
            raise ValueError(f"sum(axis_heads) ({sum(axis_heads)}) must equal num_heads ({num_heads}).")

        self.dim = dim
        self.D_size = D_size
        self.H_size = H_size
        self.W_size = W_size
        self.num_heads = num_heads
        self.axis_heads = tuple(int(heads) for heads in axis_heads)
        self.head_dim = dim // num_heads
        self.shift_enabled = bool(shift_enabled)
        self.shift_size = (int(shift_size_D), int(shift_size_H), int(shift_size_W))
        self.return_axis_desc = bool(return_axis_desc)
        self.axis_desc_mode = str(axis_desc_mode)
        if self.axis_desc_mode not in {"var_range", "mean_var_range"}:
            raise ValueError(
                "axis_desc_mode must be one of {'var_range', 'mean_var_range'}, "
                f"got {self.axis_desc_mode!r}."
            )

        d_channels = self.axis_heads[0] * self.head_dim
        h_channels = self.axis_heads[1] * self.head_dim
        w_channels = self.axis_heads[2] * self.head_dim

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.d_attention = AxisStripeAttention3D(
            d_channels,
            D_size,
            H_size,
            W_size,
            self.axis_heads[0],
            axis="D",
            stripe_window_D=stripe_window_D,
            stripe_window_H=stripe_window_H,
            stripe_window_W=stripe_window_W,
            attn_drop=attn_drop,
            shift_enabled=shift_enabled,
            shift_size_D=shift_size_D,
            shift_size_H=shift_size_H,
            shift_size_W=shift_size_W,
        )
        self.h_attention = AxisStripeAttention3D(
            h_channels,
            D_size,
            H_size,
            W_size,
            self.axis_heads[1],
            axis="H",
            stripe_window_D=stripe_window_D,
            stripe_window_H=stripe_window_H,
            stripe_window_W=stripe_window_W,
            attn_drop=attn_drop,
            shift_enabled=shift_enabled,
            shift_size_D=shift_size_D,
            shift_size_H=shift_size_H,
            shift_size_W=shift_size_W,
        )
        self.w_attention = AxisStripeAttention3D(
            w_channels,
            D_size,
            H_size,
            W_size,
            self.axis_heads[2],
            axis="W",
            stripe_window_D=stripe_window_D,
            stripe_window_H=stripe_window_H,
            stripe_window_W=stripe_window_W,
            attn_drop=attn_drop,
            shift_enabled=shift_enabled,
            shift_size_D=shift_size_D,
            shift_size_H=shift_size_H,
            shift_size_W=shift_size_W,
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_attn = nn.Parameter(torch.tensor(float(gamma_init)))

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_dim, drop=drop)
        self.gamma_mlp = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shortcut = x
        B, C, D, H, W = x.shape
        if C != self.dim:
            raise ValueError(f"Expected channel dim {self.dim}, got {C}.")
        if (D, H, W) != (self.D_size, self.H_size, self.W_size):
            raise ValueError(
                f"Expected spatial size {(self.D_size, self.H_size, self.W_size)}, got {(D, H, W)}."
            )

        x_norm = x.permute(0, 2, 3, 4, 1).contiguous()
        x_norm = self.norm1(x_norm)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, D, H, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(4, 0, 1, 2, 3, 5, 6).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        d_heads, h_heads, _ = self.axis_heads
        d_slice = slice(0, d_heads)
        h_slice = slice(d_heads, d_heads + h_heads)
        w_slice = slice(d_heads + h_heads, self.num_heads)

        out_d = self.d_attention(
            self._heads_to_channels(q[..., d_slice, :]),
            self._heads_to_channels(k[..., d_slice, :]),
            self._heads_to_channels(v[..., d_slice, :]),
        )
        out_h = self.h_attention(
            self._heads_to_channels(q[..., h_slice, :]),
            self._heads_to_channels(k[..., h_slice, :]),
            self._heads_to_channels(v[..., h_slice, :]),
        )
        out_w = self.w_attention(
            self._heads_to_channels(q[..., w_slice, :]),
            self._heads_to_channels(k[..., w_slice, :]),
            self._heads_to_channels(v[..., w_slice, :]),
        )
        axis_desc = self._compute_axis_desc(out_d, out_h, out_w) if self.return_axis_desc else None

        attn_out = torch.cat([out_d, out_h, out_w], dim=-1)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        attn_out = attn_out.permute(0, 4, 1, 2, 3).contiguous()
        x = shortcut + self.gamma_attn * self.drop_path(attn_out)

        mlp_in = x.permute(0, 2, 3, 4, 1).contiguous()
        mlp_out = self.mlp(self.norm2(mlp_in))
        mlp_out = mlp_out.permute(0, 4, 1, 2, 3).contiguous()
        x = x + self.gamma_mlp * self.drop_path(mlp_out)
        if self.return_axis_desc:
            return x, axis_desc
        return x

    def _heads_to_channels(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, heads, head_dim = x.shape
        return x.contiguous().reshape(B, D, H, W, heads * head_dim)

    def _compute_axis_desc(
        self,
        out_d: torch.Tensor,
        out_h: torch.Tensor,
        out_w: torch.Tensor,
    ) -> torch.Tensor:
        resp_d = out_d.abs().mean(dim=-1)
        resp_h = out_h.abs().mean(dim=-1)
        resp_w = out_w.abs().mean(dim=-1)
        responses = torch.stack([resp_d, resp_h, resp_w], dim=1)

        axis_mean = responses.mean(dim=1, keepdim=True)
        axis_var = responses.var(dim=1, unbiased=False, keepdim=True)
        axis_range = responses.amax(dim=1, keepdim=True) - responses.amin(dim=1, keepdim=True)
        return torch.cat([axis_mean, axis_var, axis_range], dim=1)

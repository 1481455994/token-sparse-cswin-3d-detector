from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .split_head import DropPath, Mlp


class VoxelGateIndexer3D(nn.Module):
    """Predict voxel-level suspicious scores for token-sparse stripe queries."""

    def __init__(self, channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        hidden_channels = int(hidden_channels or max(16, channels // 4))
        self.head = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


class TokenSparseAxisStripeAttention3D(nn.Module):
    """
    Axis stripe attention with sparse queries and dense keys/values per window.

    All stripe windows participate. Within each window, only the top gate-scored
    tokens are used as queries and written back, while keys and values keep the
    full stripe-window context.
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
        query_top_k_per_window: int = 16,
        min_gate_score: Optional[float] = None,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        if axis not in self.valid_axes:
            raise ValueError(f"axis must be one of {sorted(self.valid_axes)}, got {axis!r}.")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = int(dim)
        self.D_size = int(D_size)
        self.H_size = int(H_size)
        self.W_size = int(W_size)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.axis = axis
        self.query_top_k_per_window = int(query_top_k_per_window)
        self.min_gate_score = min_gate_score if min_gate_score is None else float(min_gate_score)
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        if axis == "D":
            kernel_size = (3, 1, 1)
            padding = (1, 0, 0)
            self.stripe_length = self.D_size
            self.plane_size = (self.H_size, self.W_size)
            self.plane_window = (int(stripe_window_H), int(stripe_window_W))
        elif axis == "H":
            kernel_size = (1, 3, 1)
            padding = (0, 1, 0)
            self.stripe_length = self.H_size
            self.plane_size = (self.D_size, self.W_size)
            self.plane_window = (int(stripe_window_D), int(stripe_window_W))
        else:
            kernel_size = (1, 1, 3)
            padding = (0, 0, 1)
            self.stripe_length = self.W_size
            self.plane_size = (self.D_size, self.H_size)
            self.plane_window = (int(stripe_window_D), int(stripe_window_H))

        if self.query_top_k_per_window <= 0:
            raise ValueError("query_top_k_per_window must be positive.")
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

        self.lepe_conv = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            bias=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W, C = q.shape
        if (D, H, W) != (self.D_size, self.H_size, self.W_size):
            raise ValueError(
                f"Expected spatial size {(self.D_size, self.H_size, self.W_size)}, got {(D, H, W)}."
            )
        if C != self.dim:
            raise ValueError(f"Expected channel dim {self.dim}, got {C}.")
        if tuple(gate_scores.shape) != (B, 1, D, H, W):
            raise ValueError(f"Expected gate_scores shape {(B, 1, D, H, W)}, got {tuple(gate_scores.shape)}.")

        lepe = self._compute_lepe(v)
        q = q.reshape(B, D, H, W, self.num_heads, self.head_dim)
        k = k.reshape(B, D, H, W, self.num_heads, self.head_dim)
        v = v.reshape(B, D, H, W, self.num_heads, self.head_dim)
        lepe = lepe.reshape(B, D, H, W, self.num_heads, self.head_dim)

        q = self._volume_to_axis(q)
        k = self._volume_to_axis(k)
        v = self._volume_to_axis(v)
        lepe = self._volume_to_axis(lepe)
        gate_windows = self._gate_to_axis_windows(gate_scores)

        q_windows = self._partition_axis_windows(q)
        k_windows = self._partition_axis_windows(k)
        v_windows = self._partition_axis_windows(v)
        lepe_windows = self._partition_axis_windows(lepe)

        B, win_1, win_2, heads, tokens, head_dim = q_windows.shape
        num_windows = B * win_1 * win_2
        top_k = min(self.query_top_k_per_window, tokens)

        q_windows = q_windows.reshape(num_windows, heads, tokens, head_dim)
        k_windows = k_windows.reshape(num_windows, heads, tokens, head_dim)
        v_windows = v_windows.reshape(num_windows, heads, tokens, head_dim)
        lepe_windows = lepe_windows.reshape(num_windows, heads, tokens, head_dim)
        gate_windows = gate_windows.reshape(num_windows, tokens)

        top_scores, top_indices = torch.topk(gate_windows, k=top_k, dim=-1)
        valid_queries = torch.ones_like(top_scores, dtype=torch.bool)
        if self.min_gate_score is not None:
            valid_queries = top_scores >= self.min_gate_score

        gather_index = top_indices[:, None, :, None].expand(num_windows, heads, top_k, head_dim)
        q_sparse = torch.gather(q_windows, dim=2, index=gather_index) * self.scale
        lepe_sparse = torch.gather(lepe_windows, dim=2, index=gather_index)

        attn = q_sparse @ k_windows.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out_sparse = (attn @ v_windows) + lepe_sparse
        out_sparse = out_sparse * valid_queries[:, None, :, None].to(dtype=out_sparse.dtype)

        out_windows = q_windows.new_zeros(q_windows.shape)
        out_windows.scatter_(dim=2, index=gather_index, src=out_sparse)

        query_mask_windows = torch.zeros(num_windows, tokens, dtype=torch.bool, device=q.device)
        query_mask_windows.scatter_(dim=1, index=top_indices, src=valid_queries)

        out_windows = out_windows.reshape(B, win_1, win_2, heads, tokens, head_dim)
        out = self._merge_axis_windows(out_windows)
        out = self._axis_to_volume(out)
        out = out.reshape(B, D, H, W, C)

        query_mask = query_mask_windows.reshape(B, win_1, win_2, 1, tokens, 1)
        query_mask = self._merge_axis_windows(query_mask.to(dtype=out.dtype))
        query_mask = self._axis_to_volume(query_mask)
        query_mask = query_mask.reshape(B, D, H, W).to(dtype=torch.bool)[:, None]
        return out, query_mask

    def _compute_lepe(self, v: torch.Tensor) -> torch.Tensor:
        v = v.permute(0, 4, 1, 2, 3).contiguous()
        lepe = self.lepe_conv(v)
        return lepe.permute(0, 2, 3, 4, 1).contiguous()

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
        return x.reshape(B, num_win_1, num_win_2, num_heads, tokens, head_dim)

    def _merge_axis_windows(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        window_1, window_2 = self.plane_window
        plane_1, plane_2 = self.plane_size
        num_win_1 = plane_1 // window_1
        num_win_2 = plane_2 // window_2

        x = x.reshape(
            B,
            num_win_1,
            num_win_2,
            x.shape[3],
            self.stripe_length,
            window_1,
            window_2,
            x.shape[-1],
        )
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(B, self.stripe_length, plane_1, plane_2, x.shape[-2], x.shape[-1])

    def _gate_to_axis_windows(self, gate_scores: torch.Tensor) -> torch.Tensor:
        gate = gate_scores.permute(0, 2, 3, 4, 1).unsqueeze(-1).contiguous()
        gate = self._volume_to_axis(gate)
        gate_windows = self._partition_axis_windows(gate)
        return gate_windows.squeeze(3).squeeze(-1)


class TokenSparseSplitHeadCSwinBlock3D(nn.Module):
    """Split-head 3D CSWin block with token-sparse queries inside every stripe window."""

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
        query_top_k_per_window: int = 16,
        min_gate_score: Optional[float] = None,
        mlp_ratio: float = 1.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
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
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")

        self.dim = int(dim)
        self.D_size = int(D_size)
        self.H_size = int(H_size)
        self.W_size = int(W_size)
        self.num_heads = int(num_heads)
        self.axis_heads = tuple(int(heads) for heads in axis_heads)
        self.head_dim = dim // num_heads

        d_channels = self.axis_heads[0] * self.head_dim
        h_channels = self.axis_heads[1] * self.head_dim
        w_channels = self.axis_heads[2] * self.head_dim

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.d_attention = TokenSparseAxisStripeAttention3D(
            d_channels,
            D_size,
            H_size,
            W_size,
            self.axis_heads[0],
            axis="D",
            stripe_window_D=stripe_window_D,
            stripe_window_H=stripe_window_H,
            stripe_window_W=stripe_window_W,
            query_top_k_per_window=query_top_k_per_window,
            min_gate_score=min_gate_score,
            attn_drop=attn_drop,
        )
        self.h_attention = TokenSparseAxisStripeAttention3D(
            h_channels,
            D_size,
            H_size,
            W_size,
            self.axis_heads[1],
            axis="H",
            stripe_window_D=stripe_window_D,
            stripe_window_H=stripe_window_H,
            stripe_window_W=stripe_window_W,
            query_top_k_per_window=query_top_k_per_window,
            min_gate_score=min_gate_score,
            attn_drop=attn_drop,
        )
        self.w_attention = TokenSparseAxisStripeAttention3D(
            w_channels,
            D_size,
            H_size,
            W_size,
            self.axis_heads[2],
            axis="W",
            stripe_window_D=stripe_window_D,
            stripe_window_H=stripe_window_H,
            stripe_window_W=stripe_window_W,
            query_top_k_per_window=query_top_k_per_window,
            min_gate_score=min_gate_score,
            attn_drop=attn_drop,
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, gate_scores: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        out_d, mask_d = self.d_attention(
            self._heads_to_channels(q[..., d_slice, :]),
            self._heads_to_channels(k[..., d_slice, :]),
            self._heads_to_channels(v[..., d_slice, :]),
            gate_scores,
        )
        out_h, mask_h = self.h_attention(
            self._heads_to_channels(q[..., h_slice, :]),
            self._heads_to_channels(k[..., h_slice, :]),
            self._heads_to_channels(v[..., h_slice, :]),
            gate_scores,
        )
        out_w, mask_w = self.w_attention(
            self._heads_to_channels(q[..., w_slice, :]),
            self._heads_to_channels(k[..., w_slice, :]),
            self._heads_to_channels(v[..., w_slice, :]),
            gate_scores,
        )

        write_mask = mask_d | mask_h | mask_w
        write_mask_flat = write_mask.squeeze(1).reshape(B * D * H * W)
        if not write_mask_flat.any():
            return shortcut, {"D": mask_d, "H": mask_h, "W": mask_w, "write": write_mask}

        attn_out = torch.cat([out_d, out_h, out_w], dim=-1)
        attn_flat = attn_out.reshape(B * D * H * W, C)
        proj_flat = attn_flat.new_zeros(B * D * H * W, C)
        proj_flat[write_mask_flat] = self.proj_drop(self.proj(attn_flat[write_mask_flat])).to(dtype=proj_flat.dtype)
        attn_out = proj_flat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x = shortcut + self.drop_path(attn_out)

        mlp_norm = self.norm2(x.permute(0, 2, 3, 4, 1).contiguous())
        mlp_flat = mlp_norm.reshape(B * D * H * W, C)
        mlp_out = mlp_flat.new_zeros(B * D * H * W, C)
        mlp_out[write_mask_flat] = self.mlp(mlp_flat[write_mask_flat]).to(dtype=mlp_out.dtype)
        mlp_out = mlp_out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x = x + self.drop_path(mlp_out)
        return x, {"D": mask_d, "H": mask_h, "W": mask_w, "write": write_mask}

    def _heads_to_channels(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, heads, head_dim = x.shape
        return x.contiguous().reshape(B, D, H, W, heads * head_dim)


class TokenSparseStripeCSwinAdapter3D(nn.Module):
    """
    Token-sparse CSWin adapter for the 48-scale DynUNet encoder feature.

    All D/H/W stripe windows run attention. A voxel gate selects sparse query
    tokens inside each stripe window; keys/values remain dense within the same
    window, and residual updates are written only to selected query voxels.
    """

    def __init__(
        self,
        stage_name: str,
        channels: int,
        spatial_size: int,
        query_top_k_per_window: int = 16,
        min_gate_score: Optional[float] = None,
        depth: int = 1,
        num_heads: int = 8,
        axis_head_splits: Sequence[Sequence[int]] = ((2, 3, 3), (3, 2, 3), (3, 3, 2)),
        mlp_ratio: float = 1.0,
        stripe_window_D: int = 2,
        stripe_window_H: int = 2,
        stripe_window_W: int = 2,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rates: Optional[Sequence[float]] = None,
        gamma_init: float = 0.0,
        use_soft_gate: bool = True,
        gate_hidden_channels: Optional[int] = None,
        fusion_mode: str = "residual_gate",
        fusion_conv_norm: bool = False,
        fusion_conv_act: bool = False,
    ):
        super().__init__()
        if depth < 0:
            raise ValueError(f"{stage_name} depth must be non-negative.")
        if len(axis_head_splits) == 0:
            raise ValueError("axis_head_splits must not be empty.")

        self.stage_name = stage_name
        self.channels = int(channels)
        self.spatial_size = int(spatial_size)
        self.query_top_k_per_window = int(query_top_k_per_window)
        self.min_gate_score = min_gate_score if min_gate_score is None else float(min_gate_score)
        self.use_soft_gate = bool(use_soft_gate)
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"residual_gate", "residual_concat_conv"}:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}. "
                "Expected 'residual_gate' or 'residual_concat_conv'."
            )
        self.last_sparse_stats: Dict[str, float] = {}

        self.indexer = VoxelGateIndexer3D(channels=self.channels, hidden_channels=gate_hidden_channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        if self.fusion_mode == "residual_concat_conv":
            fusion_layers = [nn.Conv3d(self.channels * 2, self.channels, kernel_size=1, bias=True)]
            if fusion_conv_norm:
                fusion_layers.append(nn.InstanceNorm3d(self.channels, affine=True))
            if fusion_conv_act:
                fusion_layers.append(nn.GELU())
            self.fusion_conv = nn.Sequential(*fusion_layers)
        else:
            self.fusion_conv = None

        num_splits = len(axis_head_splits)
        total_blocks = depth * num_splits
        if drop_path_rates is None:
            drop_path_rates = [0.0] * total_blocks
        if len(drop_path_rates) != total_blocks:
            raise ValueError(
                f"drop_path_rates length ({len(drop_path_rates)}) must equal depth * num_splits "
                f"({depth} * {num_splits} = {total_blocks})."
            )

        self.blocks = nn.ModuleList()
        block_idx = 0
        for _ in range(depth):
            for split in axis_head_splits:
                self.blocks.append(
                    TokenSparseSplitHeadCSwinBlock3D(
                        dim=self.channels,
                        D_size=self.spatial_size,
                        H_size=self.spatial_size,
                        W_size=self.spatial_size,
                        num_heads=num_heads,
                        axis_heads=tuple(int(v) for v in split),
                        stripe_window_D=stripe_window_D,
                        stripe_window_H=stripe_window_H,
                        stripe_window_W=stripe_window_W,
                        query_top_k_per_window=self.query_top_k_per_window,
                        min_gate_score=self.min_gate_score,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_rates[block_idx],
                    )
                )
                block_idx += 1

    def set_query_top_k_per_window(self, query_top_k_per_window: int):
        self.query_top_k_per_window = int(query_top_k_per_window)
        for block in self.blocks:
            block.d_attention.query_top_k_per_window = self.query_top_k_per_window
            block.h_attention.query_top_k_per_window = self.query_top_k_per_window
            block.w_attention.query_top_k_per_window = self.query_top_k_per_window

    def forward(self, main_feat: torch.Tensor):
        B, C, D, H, W = main_feat.shape
        expected_shape = (B, self.channels, self.spatial_size, self.spatial_size, self.spatial_size)
        if tuple(main_feat.shape) != expected_shape:
            raise ValueError(f"Expected {self.stage_name} feature shape {expected_shape}, got {tuple(main_feat.shape)}.")

        gate_scores = self.indexer(main_feat)
        context_feat = main_feat
        query_masks = None
        for block in self.blocks:
            context_feat, query_masks = block(context_feat, gate_scores)

        if query_masks is None:
            write_mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=main_feat.device)
            query_masks = {"D": write_mask, "H": write_mask, "W": write_mask, "write": write_mask}

        self.last_sparse_stats = self._summarize_sparse_stats(query_masks)
        sparse_residual = context_feat - main_feat
        soft_gate = gate_scores.to(dtype=main_feat.dtype) if self.use_soft_gate else query_masks["write"].to(dtype=main_feat.dtype)
        gated_residual = soft_gate * sparse_residual
        if self.fusion_conv is not None:
            fusion_input = torch.cat([main_feat, gated_residual], dim=1)
            fused_feat = main_feat + self.gamma * self.fusion_conv(fusion_input)
        else:
            fused_feat = main_feat + self.gamma * gated_residual
        return fused_feat, gate_scores.to(dtype=main_feat.dtype)

    @staticmethod
    def _safe_ratio(mask: torch.Tensor) -> float:
        return float(mask.to(dtype=torch.float32).mean().detach().cpu().item())

    def _summarize_sparse_stats(self, query_masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
        write_mask = query_masks["write"]
        return {
            "query_top_k_per_window": float(self.query_top_k_per_window),
            "d_query_ratio": self._safe_ratio(query_masks["D"]),
            "h_query_ratio": self._safe_ratio(query_masks["H"]),
            "w_query_ratio": self._safe_ratio(query_masks["W"]),
            "write_voxel_ratio": self._safe_ratio(write_mask),
            "selected_voxels": float(write_mask.reshape(write_mask.shape[0], -1).sum(dim=1).float().mean().detach().cpu()),
            "total_voxels": float(write_mask[0].numel()),
        }

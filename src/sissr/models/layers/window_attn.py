from __future__ import annotations

import math
from itertools import product
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention

from sissr.models.layers.attn import AttentionCore, _apply_rope
from sissr.models.layers.embedding import RotarySpec
from sissr.models.layers.init import (
    DEFAULT_POLICY,
    InitPolicy,
    init_attn_in,
    init_attn_out,
)

WindowSize = tuple[int, int]
ShiftSize = tuple[int, int]
MIN_LOG_TAU = math.log(0.25)
MAX_LOG_TAU = math.log(4.0)
CROSS_ATTN_CHUNK_SIZE = 64


def _format_window_size(window_size: WindowSize) -> str:
    return f"({window_size[0]}, {window_size[1]})"


def _square_window(window_size: int) -> WindowSize:
    return (window_size, window_size)


def window_partition(x: torch.Tensor, window_size: WindowSize) -> torch.Tensor:
    _, _, height, width = x.shape
    window_h, window_w = window_size
    if height % window_h != 0 or width % window_w != 0:
        raise ValueError(
            "spatial dims "
            f"({height}, {width}) must divide by window_size {_format_window_size(window_size)}"
        )
    return rearrange(
        x,
        "b c (nh wh) (nw ww) -> (b nh nw) (wh ww) c",
        wh=window_h,
        ww=window_w,
    )


def window_merge(
    x: torch.Tensor, window_size: WindowSize, height: int, width: int, batch: int
) -> torch.Tensor:
    window_h, window_w = window_size
    nh = height // window_h
    nw = width // window_w
    return rearrange(
        x,
        "(b nh nw) (wh ww) c -> b c (nh wh) (nw ww)",
        b=batch,
        nh=nh,
        nw=nw,
        wh=window_h,
        ww=window_w,
    )


def cyclic_shift(x: torch.Tensor, amount: ShiftSize) -> torch.Tensor:
    return torch.roll(x, shifts=(-amount[0], -amount[1]), dims=(2, 3))


def cyclic_unshift(x: torch.Tensor, amount: ShiftSize) -> torch.Tensor:
    return torch.roll(x, shifts=(amount[0], amount[1]), dims=(2, 3))


def _axis_slices(window_size: int, shift: int) -> tuple[slice, ...]:
    if shift == 0:
        return (slice(0, None),)
    return (
        slice(0, -window_size),
        slice(-window_size, -shift),
        slice(-shift, None),
    )


def build_shift_mask(
    height: int,
    width: int,
    window_size: WindowSize,
    shift: ShiftSize,
    device: torch.device,
) -> torch.Tensor:
    window_h, window_w = window_size
    shift_h, shift_w = shift
    if height % window_h != 0 or width % window_w != 0:
        raise ValueError(
            "spatial "
            f"({height}, {width}) not divisible by window_size {_format_window_size(window_size)}"
        )
    region_ids = torch.zeros((1, height, width, 1), device=device, dtype=torch.int32)
    h_slices = _axis_slices(window_h, shift_h)
    w_slices = _axis_slices(window_w, shift_w)
    for count, (hs, ws) in enumerate(product(h_slices, w_slices)):
        region_ids[:, hs, ws, :] = count
    region_ids = rearrange(
        region_ids,
        "b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c",
        wh=window_h,
        ww=window_w,
    ).squeeze(-1)
    return region_ids.unsqueeze(-1) != region_ids.unsqueeze(-2)


class WindowedAttention(AttentionCore, nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        *,
        num_kv_heads: int | None = None,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = True,
        gate: bool = True,
        residual_v: bool = False,
        xsa: bool = False,
        policy: InitPolicy = DEFAULT_POLICY,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.window_shape = _square_window(window_size)
        self._setup_attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            dropout=dropout,
            bias=bias,
            gate=gate,
            residual_v=residual_v,
            xsa=xsa,
            policy=policy,
        )

    def _make_shift_mask(
        self,
        height: int,
        width: int,
        shift_amount: ShiftSize,
        num_batch_windows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        bool_mask = build_shift_mask(height, width, self.window_shape, shift_amount, device=device)
        num_windows = bool_mask.shape[0]
        repeats = num_batch_windows // num_windows
        bool_mask = bool_mask.unsqueeze(0).expand(repeats, -1, -1, -1)
        bool_mask = rearrange(bool_mask, "b nw s1 s2 -> (b nw) s1 s2")
        additive = torch.zeros_like(bool_mask, dtype=dtype)
        additive.masked_fill_(bool_mask, torch.finfo(dtype).min)
        return additive.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        shift: bool = False,
        mask: torch.Tensor | None = None,
        v0: torch.Tensor | None = None,
        return_value: bool = False,
        rotary: RotarySpec | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        window_h, window_w = self.window_shape
        if height % window_h != 0 or width % window_w != 0:
            raise ValueError(
                "spatial dims "
                f"({height}, {width}) must divide by window_size "
                f"{_format_window_size(self.window_shape)}"
            )

        shift_amount = (window_h // 2, window_w // 2) if shift else (0, 0)
        if shift:
            x = cyclic_shift(x, shift_amount)

        windows = window_partition(x, self.window_shape)
        proj = self._project_qkv(windows)
        q, k = self._normalize_qk(proj.q, proj.k)
        v = proj.v

        if rotary is not None:
            q = _apply_rope(q, spec=rotary, height=window_h, width=window_w)
            k = _apply_rope(k, spec=rotary, height=window_h, width=window_w)

        v = self._blend_residual_v(v, v0)

        attn_mask: torch.Tensor | None = None
        if shift:
            attn_mask = self._make_shift_mask(
                height, width, shift_amount, windows.shape[0], dtype=q.dtype, device=q.device
            )
        if mask is not None:
            attn_mask = mask if attn_mask is None else attn_mask + mask

        q_t = rearrange(q, "b s h d -> b h s d")
        k_t = rearrange(self._expand_kv(k), "b s h d -> b h s d")
        v_t = rearrange(self._expand_kv(v), "b s h d -> b h s d")

        attn_out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        attn_out = self._apply_xsa(attn_out, v_t)
        attn_out = self._apply_gate(attn_out, windows)
        out = self._output_proj(attn_out)
        out = window_merge(out, self.window_shape, height, width, batch)

        if shift:
            out = cyclic_unshift(out, shift_amount)

        if self.use_residual_v or return_value:
            return out, v
        return out


class WindowedCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: WindowSize,
        *,
        bias: bool = True,
        policy: InitPolicy = DEFAULT_POLICY,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.left_match_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.right_match_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.left_value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.right_value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.left_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.right_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        init_attn_in(self.left_match_proj, policy)
        init_attn_in(self.right_match_proj, policy)
        init_attn_in(self.left_value_proj, policy)
        init_attn_in(self.right_value_proj, policy)
        init_attn_out(self.left_out_proj, policy)
        init_attn_out(self.right_out_proj, policy)

        self.log_tau_h = nn.Parameter(torch.zeros((num_heads,)))
        self.beta = nn.Parameter(torch.ones((1, embed_dim, 1, 1)))
        self.gamma = nn.Parameter(torch.ones((1, embed_dim, 1, 1)))

    def _row_stats(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_windows, _, seq_len, _ = query.shape
        chunk_size = min(seq_len, CROSS_ATTN_CHUNK_SIZE)
        scale = 1.0 / math.sqrt(self.head_dim)
        query_f = query.float()
        key_f = key.float()
        mask_f = attn_mask.float() if attn_mask is not None else None

        row_max = torch.full(
            (batch_windows, self.num_heads, seq_len),
            -torch.inf,
            device=query.device,
            dtype=torch.float32,
        )
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            logits = torch.matmul(query_f, key_f[:, :, start:end, :].transpose(-2, -1)) * scale
            if mask_f is not None:
                logits = logits + mask_f[:, :, :, start:end]
            row_max = torch.maximum(row_max, logits.max(dim=-1).values)

        row_sumexp = torch.zeros_like(row_max)
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            logits = torch.matmul(query_f, key_f[:, :, start:end, :].transpose(-2, -1)) * scale
            if mask_f is not None:
                logits = logits + mask_f[:, :, :, start:end]
            row_sumexp += torch.exp(logits - row_max.unsqueeze(-1)).sum(dim=-1)

        row_lse = row_max + torch.log(row_sumexp)
        return row_lse, row_max

    def _stream_cycle_confidence(
        self,
        left_q: torch.Tensor,
        right_q: torch.Tensor,
        left_k: torch.Tensor,
        right_k: torch.Tensor,
        left_attn_mask: torch.Tensor | None,
        right_attn_mask: torch.Tensor | None,
        left_row_lse: torch.Tensor,
        left_row_max: torch.Tensor,
        right_row_lse: torch.Tensor,
        right_row_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_windows, _, seq_len, _ = left_q.shape
        chunk_size = min(seq_len, CROSS_ATTN_CHUNK_SIZE)
        scale = 1.0 / math.sqrt(self.head_dim)
        left_q_f = left_q.float()
        right_q_f = right_q.float()
        left_k_f = left_k.float()
        right_k_f = right_k.float()
        left_mask_f = left_attn_mask.float() if left_attn_mask is not None else None
        right_mask_f = right_attn_mask.float() if right_attn_mask is not None else None

        left_cycle = torch.zeros_like(left_row_lse)
        right_cycle = torch.zeros_like(right_row_lse)
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)

            left_logits = (
                torch.matmul(left_q_f, right_k_f[:, :, start:end, :].transpose(-2, -1)) * scale
            )
            if left_mask_f is not None:
                left_logits = left_logits + left_mask_f[:, :, :, start:end]

            right_logits = (
                torch.matmul(right_q_f[:, :, start:end, :], left_k_f.transpose(-2, -1)) * scale
            )
            if right_mask_f is not None:
                right_logits = right_logits + right_mask_f[:, :, start:end, :]

            left_prob = torch.exp(left_logits - left_row_lse.unsqueeze(-1))
            right_prob_t = torch.exp(
                right_logits - right_row_lse[:, :, start:end].unsqueeze(-1)
            ).transpose(-2, -1)

            left_cycle += (left_prob * right_prob_t).sum(dim=-1)
            right_cycle[:, :, start:end] = (left_prob * right_prob_t).sum(dim=-2)

        left_sharp = torch.exp(left_row_max - left_row_lse)
        right_sharp = torch.exp(right_row_max - right_row_lse)
        left_conf = (left_cycle * left_sharp).mean(dim=1).clamp_(0.0, 1.0).to(dtype=left_q.dtype)
        right_conf = (
            (right_cycle * right_sharp).mean(dim=1).clamp_(0.0, 1.0).to(dtype=left_q.dtype)
        )
        return left_conf, right_conf

    def _make_shift_mask(
        self,
        height: int,
        width: int,
        runtime_window: WindowSize,
        shift_amount: ShiftSize,
        num_batch_windows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        bool_mask = build_shift_mask(height, width, runtime_window, shift_amount, device=device)
        num_windows = bool_mask.shape[0]
        repeats = num_batch_windows // num_windows
        bool_mask = bool_mask.unsqueeze(0).expand(repeats, -1, -1, -1)
        bool_mask = rearrange(bool_mask, "b nw s1 s2 -> (b nw) s1 s2")
        additive = torch.zeros_like(bool_mask, dtype=dtype)
        additive.masked_fill_(bool_mask, torch.finfo(dtype).min)
        return additive.unsqueeze(1)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        shift: bool = False,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if left.shape != right.shape:
            raise ValueError(
                "left and right inputs must have the same shape, got "
                f"{tuple(left.shape)} and {tuple(right.shape)}"
            )
        batch, _, height, width = left.shape
        full_width = self.window_size[1] == -1
        window_h = self.window_size[0]
        window_w = width if full_width else self.window_size[1]
        runtime_window: WindowSize = (window_h, window_w)
        if height % window_h != 0 or width % window_w != 0:
            raise ValueError(
                "spatial dims "
                f"({height}, {width}) must divide by window_size "
                f"{_format_window_size(runtime_window)}"
            )

        shift_w = 0 if full_width else window_w // 2
        shift_amount = (window_h // 2, shift_w) if shift else (0, 0)

        if shift:
            left = cyclic_shift(left, shift_amount)
            right = cyclic_shift(right, shift_amount)

        left_windows = window_partition(left, runtime_window)
        right_windows = window_partition(right, runtime_window)

        left_match = rearrange(
            self.left_match_proj(left_windows), "bw s (h d) -> bw s h d", h=self.num_heads
        )
        right_match = rearrange(
            self.right_match_proj(right_windows), "bw s (h d) -> bw s h d", h=self.num_heads
        )

        left_value = rearrange(
            self.left_value_proj(left_windows), "bw s (h d) -> bw s h d", h=self.num_heads
        )
        right_value = rearrange(
            self.right_value_proj(right_windows), "bw s (h d) -> bw s h d", h=self.num_heads
        )

        left_q = rearrange(left_match, "bw s h d -> bw h s d")
        right_q = rearrange(right_match, "bw s h d -> bw h s d")
        left_k = rearrange(left_match, "bw s h d -> bw h s d")
        right_k = rearrange(right_match, "bw s h d -> bw h s d")
        left_v = rearrange(left_value, "bw s h d -> bw h s d")
        right_v = rearrange(right_value, "bw s h d -> bw h s d")

        directional_mask: torch.Tensor | None = None
        if shift:
            directional_mask = self._make_shift_mask(
                height,
                width,
                runtime_window,
                shift_amount,
                left_match.shape[0],
                dtype=left_q.dtype,
                device=left_q.device,
            )

        attn_mask: torch.Tensor | None = None
        if mask is not None:
            if mask.shape[0] == left_match.shape[0]:
                directional_mask = mask if directional_mask is None else directional_mask + mask
            elif mask.shape[0] == 2 * left_match.shape[0]:
                attn_mask = mask
            else:
                raise ValueError(
                    "mask batch dimension must match either the per-direction or "
                    f"batched window count, got {mask.shape[0]}"
                )
        if directional_mask is not None:
            tiled_mask = torch.cat([directional_mask, directional_mask], dim=0)
            attn_mask = tiled_mask if attn_mask is None else tiled_mask + attn_mask
        if attn_mask is not None and attn_mask.shape[1] == 1:
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)

        both_q = torch.cat([left_q, right_q], dim=0)
        both_k = torch.cat([right_k, left_k], dim=0)
        both_v = torch.cat([right_v, left_v], dim=0)
        tau_h = torch.exp(self.log_tau_h.clamp(min=MIN_LOG_TAU, max=MAX_LOG_TAU)).to(
            dtype=both_q.dtype,
            device=both_q.device,
        )
        both_q = both_q * tau_h.view(1, self.num_heads, 1, 1)

        left_attn_mask: torch.Tensor | None = None
        right_attn_mask: torch.Tensor | None = None
        if attn_mask is not None:
            if attn_mask.shape[0] == left_q.shape[0]:
                left_attn_mask = attn_mask
                right_attn_mask = attn_mask
            else:
                left_attn_mask, right_attn_mask = attn_mask.chunk(2, dim=0)

        score_mod = None
        if attn_mask is not None:
            bias = attn_mask

            def additive_bias(
                score: torch.Tensor,
                batch: torch.Tensor,
                head: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                return score + bias[batch, head, q_idx, kv_idx]

            score_mod = additive_bias

        both_out = cast(
            torch.Tensor,
            flex_attention(
                both_q,
                both_k,
                both_v,
                score_mod=score_mod,
                scale=1.0 / math.sqrt(self.head_dim),
            ),
        )
        left_out, right_out = both_out.chunk(2, dim=0)
        left_row_lse, left_row_max = self._row_stats(
            both_q[: left_q.shape[0]], right_k, left_attn_mask
        )
        right_row_lse, right_row_max = self._row_stats(
            both_q[left_q.shape[0] :],
            left_k,
            right_attn_mask,
        )
        left_conf, right_conf = self._stream_cycle_confidence(
            left_q=both_q[: left_q.shape[0]],
            right_q=both_q[left_q.shape[0] :],
            left_k=left_k,
            right_k=right_k,
            left_attn_mask=left_attn_mask,
            right_attn_mask=right_attn_mask,
            left_row_lse=left_row_lse,
            left_row_max=left_row_max,
            right_row_lse=right_row_lse,
            right_row_max=right_row_max,
        )

        left_out = self.left_out_proj(rearrange(left_out, "bw h s d -> bw s (h d)"))
        right_out = self.right_out_proj(rearrange(right_out, "bw h s d -> bw s (h d)"))
        left_out = window_merge(left_out, runtime_window, height, width, batch)
        right_out = window_merge(right_out, runtime_window, height, width, batch)
        left_conf_map = window_merge(left_conf.unsqueeze(-1), runtime_window, height, width, batch)
        right_conf_map = window_merge(
            right_conf.unsqueeze(-1), runtime_window, height, width, batch
        )

        if shift:
            left_out = cyclic_unshift(left_out, shift_amount)
            right_out = cyclic_unshift(right_out, shift_amount)
            left_conf_map = cyclic_unshift(left_conf_map, shift_amount)
            right_conf_map = cyclic_unshift(right_conf_map, shift_amount)

        left_out = left_out * left_conf_map * self.beta
        right_out = right_out * right_conf_map * self.gamma
        return left_out, right_out

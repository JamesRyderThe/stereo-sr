from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sissr.models.enums import StereoDirection


def precompute_rotary_freqs(
    dim: int,
    max_len: int = 1000,
    theta: float = 10000.0,
    *,
    device: torch.device | str | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("rotary dimension must be even")
    idx = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    inv_freq = theta ** (-idx / dim)
    if positions is None:
        positions = torch.arange(max_len, dtype=torch.float32, device=device)
    else:
        positions = positions.to(dtype=torch.float32, device=device)
    angles = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_1d(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    rot_dim: int | None = None,
) -> torch.Tensor:
    seq_len = tensor.shape[-2]
    if rot_dim is None:
        rot_dim = freqs.shape[-1] * 2

    rot_part = tensor[..., :rot_dim]
    pass_part = tensor[..., rot_dim:]

    if positions is not None:
        freq_slice = freqs.index_select(0, positions.to(freqs.device, dtype=torch.long))
    else:
        if seq_len > freqs.shape[0]:
            raise ValueError("sequence length exceeds precomputed frequencies")
        freq_slice = freqs[:seq_len]
    freq_slice = freq_slice.to(device=rot_part.device)

    cos = freq_slice.real
    sin = freq_slice.imag
    while cos.ndim < rot_part.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    half = rot_part.shape[-1] // 2
    x_re = rot_part.float()[..., :half]
    x_im = rot_part.float()[..., half:]
    cos = cos.expand_as(x_re)
    sin = sin.expand_as(x_re)
    rotated = torch.cat((x_re * cos - x_im * sin, x_re * sin + x_im * cos), dim=-1)
    return torch.cat((rotated.to(tensor.dtype), pass_part), dim=-1)


def _expand_axis_freqs(
    axis_freqs: tuple[tuple[int, torch.Tensor], ...],
    device: torch.device,
) -> torch.Tensor:
    expanded: list[torch.Tensor] = []
    lengths = tuple(length for length, _ in axis_freqs)
    for idx, (axis_len, freq) in enumerate(axis_freqs):
        freq_slice = freq[:axis_len].to(device=device)
        view_shape = [1] * (len(lengths) + 1)
        view_shape[idx] = axis_len
        view_shape[-1] = freq_slice.shape[-1]
        freq_slice = freq_slice.view(*view_shape)
        freq_slice = freq_slice.expand(*lengths, freq_slice.shape[-1])
        expanded.append(freq_slice)
    combined = torch.cat(expanded, dim=-1)
    total = math.prod(lengths)
    return combined.reshape(total, -1)


def apply_rotary_2d(
    tensor: torch.Tensor,
    *,
    height_freqs: torch.Tensor,
    width_freqs: torch.Tensor,
    height: int,
    width: int,
    rot_dim: int | None = None,
) -> torch.Tensor:
    seq_len = tensor.shape[-2]
    total = height * width
    if total != seq_len:
        raise ValueError(f"height*width ({total}) must equal sequence length ({seq_len})")
    combined = _expand_axis_freqs(((height, height_freqs), (width, width_freqs)), tensor.device)
    return apply_rotary_1d(tensor, combined, rot_dim=rot_dim)


@dataclass(frozen=True)
class AxisConfig:
    height_pairs: int
    width_pairs: int

    @property
    def total_pairs(self) -> int:
        return self.height_pairs + self.width_pairs

    @classmethod
    def from_head_dim(
        cls,
        head_dim: int,
        *,
        share_height: float = 0.5,
        share_width: float = 0.5,
    ) -> AxisConfig:
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        total_pairs = head_dim // 2
        total_share = share_height + share_width
        raw_h = (share_height / total_share) * total_pairs
        h_pairs = max(1, round(raw_h))
        w_pairs = total_pairs - h_pairs
        if w_pairs < 1:
            w_pairs = 1
            h_pairs = total_pairs - 1
        return cls(height_pairs=h_pairs, width_pairs=w_pairs)


@dataclass(frozen=True)
class VisionGrid:
    height: int
    width: int


class SpatialRoPE(nn.Module):
    def __init__(
        self,
        axis_config: AxisConfig,
        *,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.axis_config = axis_config
        self.theta = theta
        self.rot_dim = 2 * axis_config.total_pairs
        _CacheKey = tuple[int, int, str, int | None]
        self._cache: OrderedDict[_CacheKey, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._max_cache = 16

    def _compute_freqs(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_dim = self.axis_config.height_pairs * 2
        w_dim = self.axis_config.width_pairs * 2
        pos_h = torch.arange(height, dtype=torch.float32, device=device)
        pos_w = torch.arange(width, dtype=torch.float32, device=device)
        freq_h = precompute_rotary_freqs(h_dim, theta=self.theta, device=device, positions=pos_h)
        freq_w = precompute_rotary_freqs(w_dim, theta=self.theta, device=device, positions=pos_w)
        return freq_h, freq_w

    def _get_freqs(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, device.type, device.index)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        freqs = self._compute_freqs(height, width, device)
        self._cache[key] = freqs
        while len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return freqs

    def forward(
        self,
        tensor: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if tensor.dim() != 4:
            raise ValueError("SpatialRoPE expects [batch, seq, heads, head_dim]")
        batch, seq, heads, head_dim = tensor.shape
        if head_dim < self.rot_dim:
            raise ValueError("head_dim must be at least the rotary dimension")
        freq_h, freq_w = self._get_freqs(height, width, tensor.device)
        flat = tensor.permute(0, 2, 1, 3).reshape(batch * heads, seq, head_dim)
        rotated = apply_rotary_2d(
            flat,
            height_freqs=freq_h,
            width_freqs=freq_w,
            height=height,
            width=width,
            rot_dim=self.rot_dim,
        )
        return rotated.reshape(batch, heads, seq, head_dim).permute(0, 2, 1, 3)


@dataclass(frozen=True)
class RotarySpec:
    rope: SpatialRoPE | None = None
    grid: VisionGrid | None = None

    def __post_init__(self) -> None:
        if self.rope is not None and self.grid is None:
            raise ValueError("SpatialRoPE requires a VisionGrid")
        if self.rope is None and self.grid is not None:
            raise ValueError("grid is only valid when rope is provided")

    @classmethod
    def spatial(cls, rope: SpatialRoPE, grid: VisionGrid) -> RotarySpec:
        return cls(rope=rope, grid=grid)


def _rotate_static(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().unsqueeze(0).unsqueeze(2)
    sin = angles.sin().unsqueeze(0).unsqueeze(2)
    return _apply_rotation(tensor, cos, sin, inv_freq.shape[0])


def _rotate_shifted(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    sigma: torch.Tensor,
) -> torch.Tensor:
    angles = positions.unsqueeze(-1) * inv_freq
    cos = angles.cos().unsqueeze(2)
    sin = angles.sin().unsqueeze(2)
    sigma_angles = sigma.unsqueeze(-1) * inv_freq
    dampening = torch.where(
        sigma_angles.abs() < 1e-8,
        torch.ones_like(sigma_angles),
        sigma_angles.sin() / sigma_angles,
    ).unsqueeze(2)
    return _apply_rotation(tensor, cos * dampening, sin * dampening, inv_freq.shape[0])


def _apply_rotation(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_pairs: int,
) -> torch.Tensor:
    x_re = tensor[..., :num_pairs].float()
    x_im = tensor[..., num_pairs:].float()
    rotated = torch.cat([x_re * cos - x_im * sin, x_re * sin + x_im * cos], dim=-1)
    return rotated.to(tensor.dtype)


def _validate_stereo_rope_dims(head_dim: int, height_pairs: int) -> tuple[int, int, int, int]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary encoding")
    total_pairs = head_dim // 2
    if height_pairs >= total_pairs:
        raise ValueError(
            f"height_pairs ({height_pairs}) must be less than total_pairs ({total_pairs})"
        )
    width_pairs = total_pairs - height_pairs
    return height_pairs * 2, width_pairs * 2, height_pairs, width_pairs


class EpipolarRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        embed_dim: int,
        *,
        height_pairs: int = 2,
        theta: float = 10000.0,
        init_sigma: float = 1.0,
        max_shift: float = 32.0,
    ) -> None:
        super().__init__()
        if init_sigma <= 0:
            raise ValueError("init_sigma must be positive")
        self.h_dim, self.w_dim, self.height_pairs, self.width_pairs = _validate_stereo_rope_dims(
            head_dim, height_pairs
        )
        self.head_dim = head_dim
        self.max_shift = max_shift
        self._cached_width: int = -1

        self.offset_proj = nn.Linear(embed_dim, 2, bias=True)
        nn.init.zeros_(self.offset_proj.weight)
        inv_sigma = math.log(math.expm1(init_sigma))
        with torch.no_grad():
            self.offset_proj.bias.copy_(torch.tensor([0.0, inv_sigma]))

        h_inv = theta ** (-torch.arange(0, self.h_dim, 2, dtype=torch.float32) / self.h_dim)
        w_inv = theta ** (-torch.arange(0, self.w_dim, 2, dtype=torch.float32) / self.w_dim)
        self.register_buffer("h_inv_freq", h_inv)
        self.register_buffer("w_inv_freq", w_inv)
        self.h_inv_freq: torch.Tensor
        self.w_inv_freq: torch.Tensor

    def predict_offset(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.offset_proj(tokens)
        mu: torch.Tensor = torch.tanh(raw[..., 0]) * self.max_shift
        sigma: torch.Tensor = F.softplus(raw[..., 1])
        return mu, sigma

    def _ensure_grid(self, seq_len: int, width: int, device: torch.device) -> None:
        if (
            hasattr(self, "_cached_rows")
            and self._cached_rows.shape[0] == seq_len
            and self._cached_width == width
        ):
            return
        height = seq_len // width
        rows = torch.arange(height, device=device, dtype=torch.float32).repeat_interleave(width)
        cols = torch.arange(width, device=device, dtype=torch.float32).repeat(height)
        self.register_buffer("_cached_rows", rows, persistent=False)
        self.register_buffer("_cached_cols", cols, persistent=False)
        self._cached_width = width
        self._cached_rows: torch.Tensor
        self._cached_cols: torch.Tensor

    def apply_q(self, q: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        self._ensure_grid(q.shape[1], width, q.device)
        h_part = _rotate_static(q[..., : self.h_dim], self._cached_rows, self.h_inv_freq)
        w_part = _rotate_static(q[..., self.h_dim :], self._cached_cols, self.w_inv_freq)
        return torch.cat([h_part, w_part], dim=-1)

    def apply_k(
        self,
        k: torch.Tensor,
        *,
        height: int,
        width: int,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        direction: StereoDirection = StereoDirection.LEFT_TO_RIGHT,
    ) -> torch.Tensor:
        self._ensure_grid(k.shape[1], width, k.device)
        shifted_cols = self._cached_cols.unsqueeze(0) + mu * direction
        h_part = _rotate_static(k[..., : self.h_dim], self._cached_rows, self.h_inv_freq)
        w_part = _rotate_shifted(k[..., self.h_dim :], shifted_cols, self.w_inv_freq, sigma=sigma)
        return torch.cat([h_part, w_part], dim=-1)


@dataclass(frozen=True)
class StereoGeometry:
    disparity: torch.Tensor
    sigma: torch.Tensor


class RectifiedDisparityRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        *,
        height_pairs: int = 1,
        theta: float = 10000.0,
        init_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.h_dim, self.w_dim, self.height_pairs, self.width_pairs = _validate_stereo_rope_dims(
            head_dim, height_pairs
        )
        self.head_dim = head_dim
        self._cached_width: int = -1
        self.beta_r = nn.Parameter(torch.tensor(init_beta))
        self.beta_x = nn.Parameter(torch.tensor(init_beta))

        h_inv = theta ** (-torch.arange(0, self.h_dim, 2, dtype=torch.float32) / self.h_dim)
        w_inv = theta ** (-torch.arange(0, self.w_dim, 2, dtype=torch.float32) / self.w_dim)
        self.register_buffer("h_inv_freq", h_inv)
        self.register_buffer("w_inv_freq", w_inv)
        self.h_inv_freq: torch.Tensor
        self.w_inv_freq: torch.Tensor

    def _ensure_grid(self, seq_len: int, width: int, device: torch.device) -> None:
        if (
            hasattr(self, "_cached_rows")
            and self._cached_rows.shape[0] == seq_len
            and self._cached_width == width
        ):
            return
        height = seq_len // width
        rows = torch.arange(height, device=device, dtype=torch.float32).repeat_interleave(width)
        cols = torch.arange(width, device=device, dtype=torch.float32).repeat(height)
        self.register_buffer("_cached_rows", rows, persistent=False)
        self.register_buffer("_cached_cols", cols, persistent=False)
        self._cached_width = width
        self._cached_rows: torch.Tensor
        self._cached_cols: torch.Tensor

    def apply_q(self, q: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        self._ensure_grid(q.shape[1], width, q.device)
        h_part = _rotate_static(
            q[..., : self.h_dim], self._cached_rows * self.beta_r, self.h_inv_freq
        )
        w_part = _rotate_static(
            q[..., self.h_dim :], self._cached_cols * self.beta_x, self.w_inv_freq
        )
        return torch.cat([h_part, w_part], dim=-1)

    def apply_k(
        self,
        k: torch.Tensor,
        *,
        height: int,
        width: int,
        disparity: torch.Tensor,
        sigma: torch.Tensor,
        direction: StereoDirection,
    ) -> torch.Tensor:
        self._ensure_grid(k.shape[1], width, k.device)
        projected_cols = (self._cached_cols.unsqueeze(0) + disparity * direction) * self.beta_x
        h_part = _rotate_static(
            k[..., : self.h_dim], self._cached_rows * self.beta_r, self.h_inv_freq
        )
        w_part = _rotate_shifted(
            k[..., self.h_dim :], projected_cols, self.w_inv_freq, sigma=sigma * self.beta_x
        )
        return torch.cat([h_part, w_part], dim=-1)

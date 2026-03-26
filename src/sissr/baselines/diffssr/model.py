from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sissr.models.enums import ResiConnection

_LAMBDA_INIT: Final[float] = 0.8


def _to_2tuple(x: int) -> tuple[int, int]:
    return (x, x)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError(f"num_heads ({num_heads}) must be even")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads // 2
        self.head_dim = embed_dim // num_heads
        self.scaling = qk_scale or self.head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.lambda_init = _LAMBDA_INIT
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

    def _attention_bias(
        self,
        *,
        bsz: int,
        tgt_len: int,
        src_len: int,
        rpi: Tensor,
        mask: Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        wh, ww = self.window_size
        relative_position_bias = self.relative_position_bias_table[rpi.reshape(-1)].view(
            wh * ww,
            wh * ww,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn_bias = relative_position_bias.to(device=device, dtype=dtype).unsqueeze(0)
        if mask is None:
            return attn_bias
        nw = mask.shape[0]
        if bsz % nw != 0:
            raise ValueError(f"batch size {bsz} must be divisible by attention mask windows {nw}")
        mask_bias = mask.to(device=device, dtype=dtype)
        attn_bias = attn_bias.view(1, 1, 2 * self.num_heads, tgt_len, src_len) + mask_bias.view(
            1,
            nw,
            1,
            tgt_len,
            src_len,
        )
        attn_bias = attn_bias.expand(bsz // nw, nw, 2 * self.num_heads, tgt_len, src_len)
        return attn_bias.reshape(bsz, 2 * self.num_heads, tgt_len, src_len).contiguous()

    def forward(self, x: Tensor, rpi: Tensor, mask: Tensor | None = None) -> Tensor:
        bsz, tgt_len, _ = x.size()
        src_len = tgt_len

        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = self.k_proj(x).view(bsz, src_len, self.num_heads, 2, self.head_dim)
        v = self.v_proj(x).view(bsz, src_len, self.num_heads, 2 * self.head_dim)

        q = q.permute(0, 2, 3, 1, 4).reshape(bsz, 2 * self.num_heads, tgt_len, self.head_dim)
        k = k.permute(0, 2, 3, 1, 4).reshape(bsz, 2 * self.num_heads, src_len, self.head_dim)
        v = v.permute(0, 2, 1, 3).contiguous()
        v_sdpa = v[:, :, None, :, :].expand(bsz, self.num_heads, 2, src_len, 2 * self.head_dim)
        v_sdpa = v_sdpa.reshape(bsz, 2 * self.num_heads, src_len, 2 * self.head_dim).contiguous()

        attn_bias = self._attention_bias(
            bsz=bsz,
            tgt_len=tgt_len,
            src_len=src_len,
            rpi=rpi,
            mask=mask,
            dtype=q.dtype,
            device=q.device,
        )
        attn = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v_sdpa,
            attn_mask=attn_bias,
            dropout_p=0.0,
            scale=self.scaling,
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn.view(bsz, self.num_heads, 2, tgt_len, 2 * self.head_dim)
        attn = attn[:, :, 0] - lambda_full * attn[:, :, 1]
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)

        attn = self.attn_drop(attn)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)
        result: Tensor = attn
        return result


def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob or 0.0

    def forward(self, x: Tensor) -> Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _window_partition(x: Tensor, window_size: tuple[int, int]) -> Tensor:
    b, h, w, c = x.shape
    wh, ww = window_size
    x = x.view(b, h // wh, wh, w // ww, ww, c)
    windows: Tensor = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, wh, ww, c)
    return windows


def _window_reverse(
    windows: Tensor,
    window_size: tuple[int, int],
    h: int,
    w: int,
) -> Tensor:
    wh, ww = window_size
    b = windows.shape[0] // (h // wh * w // ww)
    x = windows.view(b, h // wh, w // ww, wh, ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, h, w, -1)
    return x


class SSCAMWindow(nn.Module):
    def __init__(
        self,
        c: int,
        window_size: int = 3,
        shift_size: int = 0,
    ) -> None:
        super().__init__()
        self.scale = c**-0.5
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm = nn.LayerNorm(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        h, w = x_size
        b, _, c = x.shape
        win_size = self.window_size

        x = self.norm(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        x_l = x[: b // 2]
        x_r = x[b // 2 :]

        if self.shift_size > 0:
            x_l = torch.roll(x_l, shifts=-self.shift_size, dims=2)
            x_r = torch.roll(x_r, shifts=-self.shift_size, dims=2)

        q_l = self.l_proj1(x_l).permute(0, 2, 3, 1).contiguous()
        q_r = self.r_proj1(x_r).permute(0, 2, 3, 1).contiguous()
        q_l_win = _window_partition(q_l, (win_size, w))
        q_l_win = q_l_win.view(-1, win_size * w, c)
        q_r_win = _window_partition(q_r, (win_size, w))
        q_r_win = q_r_win.view(-1, win_size * w, c)
        q_r_win = q_r_win.permute(0, 2, 1).contiguous()

        attention = torch.matmul(q_l_win, q_r_win) * self.scale

        v_l = self.l_proj2(x_l).permute(0, 2, 3, 1).contiguous()
        v_r = self.r_proj2(x_r).permute(0, 2, 3, 1).contiguous()
        v_l_win = _window_partition(v_l, (win_size, w)).view(-1, win_size * w, c)
        v_r_win = _window_partition(v_r, (win_size, w)).view(-1, win_size * w, c)

        f_r2l = torch.matmul(torch.softmax(attention, dim=-1), v_r_win)
        f_l2r = torch.matmul(
            torch.softmax(attention.permute(0, 2, 1).contiguous(), dim=-1),
            v_l_win,
        )

        f_r2l = f_r2l.view(-1, win_size, w, c)
        f_l2r = f_l2r.view(-1, win_size, w, c)

        f_r2l = _window_reverse(f_r2l, (win_size, w), h, w).permute(0, 3, 1, 2)
        f_l2r = _window_reverse(f_l2r, (win_size, w), h, w).permute(0, 3, 1, 2)

        out_l = x_l + f_r2l * self.beta
        out_r = x_r + f_l2r * self.gamma

        if self.shift_size > 0:
            out_l = torch.roll(out_l, shifts=self.shift_size, dims=2)
            out_r = torch.roll(out_r, shifts=self.shift_size, dims=2)

        out = torch.cat([out_l, out_r], 0)
        out = out.flatten(2).transpose(1, 2)
        return out


class DCAL(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        i_layer: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = MultiheadDiffAttn(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        if i_layer % 2 == 0:
            self.scam_layer = SSCAMWindow(dim, 4)
        else:
            self.scam_layer = SSCAMWindow(dim, 4, 4 // 2)

        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: Tensor,
        x_size: tuple[int, int],
        rpi_sa: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        h, w = x_size
        b, _, c = x.shape

        window_size, shift_size = self.window_size, self.shift_size
        if min(x_size) <= window_size:
            shift_size = 0
            window_size = min(x_size)

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            mask = attn_mask
        else:
            shifted_x = x
            mask = None

        x_windows = _window_partition(shifted_x, (window_size, window_size))
        x_windows = x_windows.view(-1, window_size * window_size, c)

        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=mask)

        attn_windows = attn_windows.view(-1, window_size, window_size, c)
        shifted_x = _window_reverse(attn_windows, (window_size, window_size), h, w)

        if shift_size > 0:
            attn_x = torch.roll(
                shifted_x,
                shifts=(shift_size, shift_size),
                dims=(1, 2),
            )
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        x = shortcut + self.drop_path(attn_x)
        x = self.scam_layer(x, x_size)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AttenBlocks(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                DCAL(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    norm_layer=norm_layer,
                    i_layer=i,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample: nn.Module | None = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(
        self,
        x: Tensor,
        x_size: tuple[int, int],
        params: dict[str, Tensor | None],
    ) -> Tensor:
        rpi_sa = params["rpi_sa"]
        attn_mask = params["attn_mask"]
        assert rpi_sa is not None
        for blk in self.blocks:
            assert isinstance(blk, DCAL)
            x = blk(x, x_size, rpi_sa, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class DCAB(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
        patch_size: int = 4,
        resi_connection: ResiConnection = ResiConnection.ONE_CONV,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.residual_group = AttenBlocks(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
        )

        if resi_connection == ResiConnection.ONE_CONV:
            self.conv: nn.Module = nn.Conv2d(dim, dim, 3, 1, 1)
        else:
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None
        )

        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, in_chans=0, embed_dim=dim)

    def forward(
        self,
        x: Tensor,
        x_size: tuple[int, int],
        params: dict[str, Tensor | None],
    ) -> Tensor:
        result: Tensor = (
            self.patch_embed(
                self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))
            )
            + x
        )
        return result


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.LayerNorm] | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm: nn.LayerNorm | None = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
    ) -> None:
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:
        return (
            x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        )


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        m: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        super().__init__(*m)


class DIFFSSR(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        patch_norm: bool = True,
        upscale: int = 2,
        img_range: float = 1.0,
        resi_connection: ResiConnection = ResiConnection.ONE_CONV,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            mean = torch.tensor(rgb_mean, dtype=torch.float32).view(1, 3, 1, 1)
        else:
            mean = torch.zeros(1, 1, 1, 1)
        self.register_buffer("mean", mean)
        self.mean: Tensor
        self.upscale = upscale

        relative_position_index_sa = self._calculate_rpi_sa()
        self.register_buffer("relative_position_index_SA", relative_position_index_sa)
        self.relative_position_index_SA: Tensor

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.patch_unembed = PatchUnEmbed(
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DCAB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _calculate_rpi_sa(self) -> Tensor:
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index: Tensor = relative_coords.sum(-1)
        return relative_position_index

    def _calculate_mask(self, x_size: tuple[int, int]) -> Tensor:
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = _window_partition(img_mask, (self.window_size, self.window_size))
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set[str]:
        return {"relative_position_bias_table"}

    def forward_features(self, x: Tensor) -> Tensor:
        x_size = (x.shape[2], x.shape[3])

        attn_mask = self._calculate_mask(x_size).to(x.device)
        params: dict[str, Tensor | None] = {
            "attn_mask": attn_mask,
            "rpi_sa": self.relative_position_index_SA,
        }

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            assert isinstance(layer, DCAB)
            x = layer(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def _pad_to_window(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        x_l = x[:, : c // 2]
        x_r = x[:, c // 2 :]
        orig_h, orig_w = x_l.shape[2], x_l.shape[3]
        x = torch.cat([self._pad_to_window(x_l), self._pad_to_window(x_r)], 0)

        mean = self.mean.to(x)
        x = (x - mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + mean

        x = torch.cat([x[:b], x[b:]], 1)
        return x[:, :, : orig_h * self.upscale, : orig_w * self.upscale]

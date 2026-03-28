from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from sissr.models.enums import ResidualStrategy
from sissr.models.layers.window_attn import WindowedCrossAttention
from sissr.models.stereo_sr import StereoBody, StereoSRModel


def _make_model(
    window_size: int = 4,
    *,
    cross_window_size: tuple[int, int] = (4, 8),
    residual_strategy: ResidualStrategy = ResidualStrategy.DEPTH_AGG,
    blocks_per_group: int | None = None,
) -> StereoSRModel:
    return StereoSRModel(
        embed_dim=48,
        num_heads=4,
        num_blocks=4,
        window_size=window_size,
        cross_window_size=cross_window_size,
        residual_strategy=residual_strategy,
        mlp_hidden_dim=64,
        blocks_per_group=blocks_per_group,
        upscale=4,
        img_range=1.0,
    )


@pytest.mark.parametrize(
    "residual_strategy",
    [ResidualStrategy.DEPTH_AGG, ResidualStrategy.STANDARD],
)
def test_forward_pass_shape_and_finite(residual_strategy: ResidualStrategy) -> None:
    model = _make_model(residual_strategy=residual_strategy)
    model.eval()
    x = torch.randn(1, 6, 16, 16)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 64, 64)
    assert out.isfinite().all()


def test_full_width_cross_attention_forward_shape_and_finite() -> None:
    model = _make_model(cross_window_size=(4, -1))
    model.eval()
    x = torch.randn(1, 6, 16, 16)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 64, 64)
    assert out.isfinite().all()


def test_full_width_cross_attention_all_params_grad() -> None:
    model = _make_model(cross_window_size=(4, -1))
    model.train()
    x = torch.randn(1, 6, 8, 8)
    out: Tensor = model(x)
    loss = out.sum()
    torch.autograd.backward(loss)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for {name}"


def test_nonsquare_input() -> None:
    model = _make_model()
    model.eval()
    x = torch.randn(1, 6, 16, 24)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 64, 96)
    assert out.isfinite().all()


def test_cross_window_padding_shape_and_finite() -> None:
    model = _make_model()
    model.eval()
    x = torch.randn(1, 6, 16, 20)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 64, 80)
    assert out.isfinite().all()


def test_cross_window_size_changes_output_with_same_weights() -> None:
    torch.manual_seed(0)
    bounded = _make_model(cross_window_size=(4, 8))
    full_width = _make_model(cross_window_size=(4, -1))
    full_width.load_state_dict(bounded.state_dict())
    bounded.eval()
    full_width.eval()
    x = torch.randn(1, 6, 16, 16)

    with torch.no_grad():
        out_bounded: Tensor = bounded(x)
        out_full_width: Tensor = full_width(x)

    assert torch.max(torch.abs(out_bounded - out_full_width)).item() > 0.0


def test_cross_view_transfer_gains_change_model_output() -> None:
    torch.manual_seed(0)
    enabled = _make_model(residual_strategy=ResidualStrategy.STANDARD)
    disabled = _make_model(residual_strategy=ResidualStrategy.STANDARD)
    disabled.load_state_dict(enabled.state_dict())
    enabled.eval()
    disabled.eval()

    with torch.no_grad():
        for module in disabled.modules():
            if isinstance(module, WindowedCrossAttention):
                module.beta.zero_()
                module.gamma.zero_()

    x = torch.randn(1, 6, 16, 16)
    with torch.no_grad():
        enabled_out = enabled(x)
        disabled_out = disabled(x)

    assert torch.max(torch.abs(enabled_out - disabled_out)).item() > 0.0


def test_output_dtype_matches_input() -> None:
    model = _make_model()
    model.eval()
    x = torch.randn(1, 6, 16, 16)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.dtype == x.dtype


def test_invalid_input_channels_raise() -> None:
    model = _make_model()
    x = torch.randn(1, 3, 16, 16)
    with pytest.raises(ValueError, match="expects 6-channel stereo input"):
        model(x)


def test_unsupported_upscale_raises() -> None:
    with pytest.raises(ValueError, match="scale 5 is not supported"):
        StereoSRModel(
            embed_dim=48,
            num_heads=4,
            num_blocks=2,
            window_size=4,
            cross_window_size=(4, 8),
            mlp_hidden_dim=64,
            upscale=5,
        )


def test_all_parameters_receive_gradients() -> None:
    model = _make_model()
    model.train()
    x = torch.randn(1, 6, 8, 8)
    out: Tensor = model(x)
    loss = out.sum()
    torch.autograd.backward(loss)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for {name}"


def test_per_sublayer_aggregate_count() -> None:
    from sissr.models.layers.depth_agg import DepthState

    model = _make_model()
    model.eval()

    orig = DepthState.aggregate
    counts: list[int] = [0]

    def counting(self: DepthState) -> Tensor:
        counts[0] += 1
        return orig(self)

    DepthState.aggregate = counting  # type: ignore[assignment]
    try:
        with torch.no_grad():
            model(torch.randn(1, 6, 16, 16))
    finally:
        DepthState.aggregate = orig  # type: ignore[assignment]

    num_blocks = 4
    expected = 2 * 3 * num_blocks
    assert counts[0] == expected, f"expected {expected} aggregates, got {counts[0]}"


def test_depth_agg_allocates_finalize_query() -> None:
    model = _make_model()
    assert isinstance(model.body, StereoBody)

    expected_query_count = 3 * len(model.body.blocks) + 1
    assert model.body.left_depth.queries.shape == (expected_query_count, 48)
    assert model.body.right_depth.queries.shape == (expected_query_count, 48)


def test_depth_agg_defaults_blocks_per_group_to_8() -> None:
    model = _make_model()
    assert isinstance(model.body, StereoBody)

    assert model.body.blocks_per_group == 8


def test_depth_agg_blocks_per_group_sets_grouped_source_budget() -> None:
    model = _make_model(blocks_per_group=2)
    assert isinstance(model.body, StereoBody)

    expected_max_sources = math.ceil(len(model.body.blocks) / model.body.blocks_per_group) + 1
    assert model.body.left_depth.max_sources == expected_max_sources
    assert model.body.right_depth.max_sources == expected_max_sources


def test_depth_agg_blocks_per_group_reduces_commit_boundaries() -> None:
    from sissr.models.layers.depth_agg import DepthState

    model = _make_model(blocks_per_group=2)
    model.eval()
    assert isinstance(model.body, StereoBody)

    orig = DepthState.commit_boundary
    counts: list[int] = [0]

    def counting(self: DepthState) -> None:
        counts[0] += 1
        orig(self)

    DepthState.commit_boundary = counting  # type: ignore[assignment]
    try:
        with torch.no_grad():
            out: Tensor = model(torch.randn(1, 6, 16, 16))
    finally:
        DepthState.commit_boundary = orig  # type: ignore[assignment]

    expected = 2 * math.ceil(len(model.body.blocks) / model.body.blocks_per_group)
    assert counts[0] == expected
    assert out.shape == (1, 6, 64, 64)
    assert out.isfinite().all()


def test_depth_agg_blocks_per_group_gt_num_blocks_is_single_group() -> None:
    model = _make_model(blocks_per_group=8)
    assert isinstance(model.body, StereoBody)

    assert model.body.left_depth.max_sources == 2
    assert model.body.right_depth.max_sources == 2


def test_standard_residual_nonsquare() -> None:
    model = _make_model(residual_strategy=ResidualStrategy.STANDARD)
    model.eval()
    x = torch.randn(1, 6, 16, 24)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 64, 96)
    assert out.isfinite().all()


def test_standard_residual_all_parameters_receive_gradients() -> None:
    model = _make_model(residual_strategy=ResidualStrategy.STANDARD)
    model.train()
    x = torch.randn(1, 6, 8, 8)
    out: Tensor = model(x)
    loss = out.sum()
    torch.autograd.backward(loss)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for {name}"


def test_standard_residual_no_depth_aggregator_params() -> None:
    model = _make_model(residual_strategy=ResidualStrategy.STANDARD)
    for name, _ in model.named_parameters():
        assert "depth" not in name, f"unexpected depth aggregator param: {name}"


def test_standard_residual_rejects_blocks_per_group() -> None:
    with pytest.raises(ValueError, match="blocks_per_group requires residual_strategy"):
        _make_model(residual_strategy=ResidualStrategy.STANDARD, blocks_per_group=2)


def test_depth_agg_rejects_nonpositive_blocks_per_group() -> None:
    with pytest.raises(ValueError, match="blocks_per_group must be positive"):
        _make_model(blocks_per_group=0)


def test_standard_residual_no_aggregate_calls() -> None:
    from sissr.models.layers.depth_agg import DepthState

    model = _make_model(residual_strategy=ResidualStrategy.STANDARD)
    model.eval()

    orig = DepthState.aggregate
    counts: list[int] = [0]

    def counting(self: DepthState) -> Tensor:
        counts[0] += 1
        return orig(self)

    DepthState.aggregate = counting  # type: ignore[assignment]
    try:
        with torch.no_grad():
            model(torch.randn(1, 6, 16, 16))
    finally:
        DepthState.aggregate = orig  # type: ignore[assignment]

    assert counts[0] == 0, f"expected 0 aggregates, got {counts[0]}"

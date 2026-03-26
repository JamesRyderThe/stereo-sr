from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sissr.configs.loading import load_config
from sissr.configs.schema import (
    BatchMixMode,
    DataConfig,
    DiffSSRModelConfig,
    ExperimentConfig,
    LossConfig,
    StereoSRModelConfig,
    TrainConfig,
)
from sissr.models.enums import CrossPosEncoding, ResidualStrategy


def _write_yaml(tmp_path: Path, content: str) -> str:
    p = tmp_path / "test.yaml"
    p.write_text(content)
    return str(p)


def test_valid_yaml_loads(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "experiment_name: test_run\nmodel:\n  embed_dim: 180\n",
    )
    config = load_config(yaml_path)
    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.model, DiffSSRModelConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.train, TrainConfig)
    assert isinstance(config.loss, LossConfig)
    assert config.experiment_name == "test_run"
    assert config.model.embed_dim == 180


def test_missing_required_field_uses_default(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, "experiment_name: minimal\n")
    config = load_config(yaml_path)
    assert config.model.embed_dim == 180
    assert config.model.num_blocks == 13
    assert config.data.scale == 4
    assert config.data.augment.horizontal_shift_max_px == 8
    assert config.data.augment.batch_mix_mode == BatchMixMode.MIXUP
    assert config.train.seed == 42


def test_scale_upscale_mismatch_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "data:\n  scale: 2\nmodel:\n  upscale: 4\n",
    )
    with pytest.raises(ValidationError, match="data.scale.*!=.*model.upscale"):
        load_config(yaml_path)


def test_patch_lr_not_divisible_by_window_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "data:\n  patch_size_lr: [30, 90]\nmodel:\n  window_size: 16\n",
    )
    with pytest.raises(ValidationError, match="divisible by.*window_size"):
        load_config(yaml_path)


def test_patch_lr_height_not_divisible_by_4_raises(tmp_path: Path) -> None:
    yaml_content = (
        "data:\n  patch_size_lr: [18, 96]\n"
        "model:\n  window_size: 6\n  num_heads: 6\n  embed_dim: 180\n"
    )
    yaml_path = _write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValidationError, match="divisible by 4"):
        load_config(yaml_path)


def test_odd_num_heads_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n  num_heads: 5\n  embed_dim: 180\n",
    )
    with pytest.raises(ValidationError, match="num_heads.*must be even"):
        load_config(yaml_path)


def test_stereo_sr_residual_strategy_default() -> None:
    config = StereoSRModelConfig(
        embed_dim=48,
        num_heads=4,
        num_blocks=2,
        window_size=4,
        cross_window_size=(4, 8),
    )
    assert config.residual_strategy == ResidualStrategy.DEPTH_AGG


def test_stereo_sr_defaults_override_base_model_defaults() -> None:
    config = StereoSRModelConfig()

    assert config.embed_dim == 192
    assert config.num_blocks == 36


def test_stereo_sr_blocks_per_group_allowed_for_depth_agg() -> None:
    config = StereoSRModelConfig(
        embed_dim=48,
        num_heads=4,
        num_blocks=4,
        window_size=4,
        cross_window_size=(4, 8),
        blocks_per_group=2,
    )

    assert config.blocks_per_group == 2


def test_stereo_sr_blocks_per_group_one_allowed_for_standard() -> None:
    config = StereoSRModelConfig(
        embed_dim=48,
        num_heads=4,
        num_blocks=4,
        window_size=4,
        cross_window_size=(4, 8),
        residual_strategy=ResidualStrategy.STANDARD,
        blocks_per_group=1,
    )

    assert config.blocks_per_group == 1


def test_stereo_sr_blocks_per_group_rejected_for_standard() -> None:
    with pytest.raises(ValidationError, match="blocks_per_group requires residual_strategy"):
        StereoSRModelConfig(
            embed_dim=48,
            num_heads=4,
            num_blocks=4,
            window_size=4,
            cross_window_size=(4, 8),
            residual_strategy=ResidualStrategy.STANDARD,
            blocks_per_group=2,
        )


def test_stereo_sr_build_passes_blocks_per_group() -> None:
    from sissr.models.stereo_sr import StereoBody

    config = StereoSRModelConfig(
        embed_dim=48,
        num_heads=4,
        num_blocks=4,
        window_size=4,
        cross_window_size=(4, 8),
        blocks_per_group=2,
    )

    model = config.build()

    assert isinstance(model.body, StereoBody)
    assert model.body.blocks_per_group == 2


def test_stereo_sr_build_defaults_blocks_per_group_to_8() -> None:
    from sissr.models.stereo_sr import StereoBody

    config = StereoSRModelConfig(
        embed_dim=48,
        num_heads=4,
        num_blocks=4,
        window_size=4,
        cross_window_size=(4, 8),
    )

    model = config.build()

    assert isinstance(model.body, StereoBody)
    assert model.body.blocks_per_group == 8


def test_stereo_sr_ablation_configs_remain_distinct() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ablation_config = load_config(str(repo_root / "configs/ablation_stereo_sr.yaml"))
    rect_none_config = load_config(str(repo_root / "configs/ablation_stereo_sr_rect_none.yaml"))

    assert isinstance(ablation_config.model, StereoSRModelConfig)
    assert isinstance(rect_none_config.model, StereoSRModelConfig)
    assert (
        ablation_config.model.cross_window_size,
        ablation_config.model.cross_pos_encoding,
        ablation_config.model.residual_strategy,
        ablation_config.model.blocks_per_group,
    ) != (
        rect_none_config.model.cross_window_size,
        rect_none_config.model.cross_pos_encoding,
        rect_none_config.model.residual_strategy,
        rect_none_config.model.blocks_per_group,
    )


def test_stereo_sr_full_width_cross_window_accepted() -> None:
    config = StereoSRModelConfig(
        embed_dim=48,
        num_heads=4,
        num_blocks=4,
        window_size=4,
        cross_window_size=(4, -1),
        cross_pos_encoding=CrossPosEncoding.NONE,
    )
    assert config.cross_window_size == (4, -1)


def test_stereo_sr_full_width_requires_none_encoding() -> None:
    for encoding in [
        CrossPosEncoding.WINDOW_ROPE,
        CrossPosEncoding.EPIPOLAR_ROPE,
        CrossPosEncoding.RECTIFIED_DISPARITY_ROPE,
    ]:
        with pytest.raises(ValidationError, match="full-width"):
            StereoSRModelConfig(
                embed_dim=48,
                num_heads=4,
                num_blocks=4,
                window_size=4,
                cross_window_size=(4, -1),
                cross_pos_encoding=encoding,
            )


def test_stereo_sr_odd_num_heads_allowed_when_head_dim_even(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n"
        "  name: stereo_sr\n"
        "  embed_dim: 30\n"
        "  num_heads: 3\n"
        "  window_size: 6\n"
        "  cross_window_size: [6, 12]\n"
        "data:\n"
        "  patch_size_lr: [18, 96]\n",
    )

    config = load_config(yaml_path)

    assert isinstance(config.model, StereoSRModelConfig)
    assert config.model.num_heads == 3


def test_stereo_sr_odd_head_dim_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n"
        "  name: stereo_sr\n"
        "  embed_dim: 39\n"
        "  num_heads: 3\n"
        "  window_size: 6\n"
        "  cross_window_size: [6, 12]\n"
        "data:\n"
        "  patch_size_lr: [18, 96]\n",
    )

    with pytest.raises(ValidationError, match="head_dim.*must be even"):
        load_config(yaml_path)


def test_stereo_sr_odd_cross_window_axis_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n  name: stereo_sr\n  window_size: 16\n  cross_window_size: [7, 32]\n",
    )

    with pytest.raises(ValidationError, match="cross_window_size.*must be even"):
        load_config(yaml_path)


def test_stereo_sr_cross_window_axis_too_small_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n  name: stereo_sr\n  window_size: 16\n  cross_window_size: [0, 32]\n",
    )

    with pytest.raises(ValidationError, match="cross_window_size.*at least 2"):
        load_config(yaml_path)


def test_stereo_sr_patch_lr_respects_combined_window_divisibility(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n"
        "  name: stereo_sr\n"
        "  window_size: 16\n"
        "  cross_window_size: [8, 32]\n"
        "data:\n"
        "  patch_size_lr: [32, 100]\n",
    )

    with pytest.raises(ValidationError, match="combined stereo windows"):
        load_config(yaml_path)


def test_stereo_sr_patch_lr_height_not_divisible_by_4_is_allowed(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n"
        "  name: stereo_sr\n"
        "  window_size: 6\n"
        "  cross_window_size: [6, 12]\n"
        "data:\n"
        "  patch_size_lr: [18, 96]\n",
    )

    config = load_config(yaml_path)

    assert isinstance(config.model, StereoSRModelConfig)


def test_diffssr_stereo_only_fields_raise(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n  name: diffssr\n  cross_window_size: [8, 32]\n",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(yaml_path)


def test_stereo_sr_diffssr_only_fields_raise(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n  name: stereo_sr\n  block_depth: 2\n",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(yaml_path)


def test_cli_override_merging(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, "data:\n  batch_size: 4\n")
    config = load_config(yaml_path, overrides=["data.batch_size=8"])
    assert config.data.batch_size == 8


def test_nested_augment_override_merging(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "data:\n  augment:\n    horizontal_shift_prob: 0.25\n",
    )
    config = load_config(
        yaml_path,
        overrides=["data.augment.horizontal_shift_max_px=4", "data.augment.batch_mix_mode=off"],
    )

    assert config.data.augment.horizontal_shift_prob == 0.25
    assert config.data.augment.horizontal_shift_max_px == 4
    assert config.data.augment.batch_mix_mode == BatchMixMode.OFF


def test_sequence_override_merging(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n  name: stereo_sr\n",
    )
    config = load_config(
        yaml_path,
        overrides=["model.cross_window_size=[4, 8]", "data.patch_size_lr=[32, 96]"],
    )

    assert isinstance(config.model, StereoSRModelConfig)
    assert config.model.cross_window_size == (4, 8)
    assert config.data.patch_size_lr == (32, 96)


def test_cli_override_parses_boolean_and_enum_values(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "data:\n  augment:\n    batch_mix_mode: off\n",
    )
    config = load_config(
        yaml_path,
        overrides=["data.use_geometry=true", "data.augment.batch_mix_mode=off"],
    )

    assert config.data.use_geometry is True
    assert config.data.augment.batch_mix_mode == BatchMixMode.OFF


def test_prefetch_factor_override_merging(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, "data:\n  prefetch_factor: 2\n")
    config = load_config(yaml_path, overrides=["data.prefetch_factor=4"])
    assert config.data.prefetch_factor == 4


def test_unknown_override_field_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, "data:\n  batch_size: 4\n")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(yaml_path, overrides=["data.typo_field=123"])


def test_geometry_with_batch_mix_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "data:\n  use_geometry: true\n",
    )

    with pytest.raises(ValidationError, match="batch_mix_mode must be 'off'"):
        load_config(yaml_path)


def test_embed_dim_not_divisible_by_num_heads_raises(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "model:\n  embed_dim: 180\n  num_heads: 8\n",
    )
    with pytest.raises(ValidationError, match="embed_dim.*divisible by.*num_heads"):
        load_config(yaml_path)


def test_default_config_is_valid() -> None:
    ExperimentConfig()

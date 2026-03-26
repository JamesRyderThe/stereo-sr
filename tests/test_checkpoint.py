from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import Accelerator
from torch import nn

from sissr.train.checkpoint import CheckpointManager


def _make_accelerator() -> Accelerator:
    return Accelerator(cpu=True)


def test_save_creates_dir_with_metadata(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    mgr = CheckpointManager(str(tmp_path / "ckpts"), max_checkpoints=3)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters())
    accelerator.prepare(model, optimizer)

    mgr.save(accelerator, step=100, metric=0.5)

    step_dir = tmp_path / "ckpts" / "step_00000100"
    assert step_dir.exists()
    metadata = json.loads((step_dir / "metadata.json").read_text())
    assert metadata["step"] == 100
    assert metadata["metric"] == 0.5


def test_find_latest_none_when_empty(tmp_path: Path) -> None:
    mgr = CheckpointManager(str(tmp_path / "ckpts"))
    assert mgr.find_latest() is None


def test_find_latest_returns_path_after_save(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    mgr = CheckpointManager(str(tmp_path / "ckpts"))
    model = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters())
    accelerator.prepare(model, optimizer)

    mgr.save(accelerator, step=200)

    latest = mgr.find_latest()
    assert latest is not None
    assert latest.name == "step_00000200"


def test_rotation_keeps_max_checkpoints(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    mgr = CheckpointManager(str(tmp_path / "ckpts"), max_checkpoints=3)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters())
    accelerator.prepare(model, optimizer)

    for i in range(5):
        mgr.save(accelerator, step=(i + 1) * 100)

    ckpt_dir = tmp_path / "ckpts"
    step_dirs = sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_"))
    assert len(step_dirs) == 3
    assert step_dirs[0].name == "step_00000300"
    assert step_dirs[1].name == "step_00000400"
    assert step_dirs[2].name == "step_00000500"


def test_best_updates_on_improvement(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    mgr = CheckpointManager(str(tmp_path / "ckpts"))
    model = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters())
    accelerator.prepare(model, optimizer)

    mgr.save(accelerator, step=100, metric=20.0)
    mgr.save(accelerator, step=200, metric=25.0)

    best_link = tmp_path / "ckpts" / "best"
    assert best_link.is_symlink()
    assert best_link.resolve().name == "step_00000200"


def test_best_stable_on_worse_metric(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    mgr = CheckpointManager(str(tmp_path / "ckpts"))
    model = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters())
    accelerator.prepare(model, optimizer)

    mgr.save(accelerator, step=100, metric=25.0)
    mgr.save(accelerator, step=200, metric=20.0)

    best_link = tmp_path / "ckpts" / "best"
    assert best_link.resolve().name == "step_00000100"


def test_load_step_returns_correct_value(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    mgr = CheckpointManager(str(tmp_path / "ckpts"))
    model = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters())
    accelerator.prepare(model, optimizer)

    mgr.save(accelerator, step=1000, metric=22.5)

    latest = mgr.find_latest()
    assert latest is not None
    assert mgr.load_step(latest) == 1000


def test_accelerator_round_trip(tmp_path: Path) -> None:
    accelerator = _make_accelerator()
    model = nn.Linear(4, 2, bias=False)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer = accelerator.prepare(model, optimizer)

    original_weight = accelerator.unwrap_model(model).weight.data.clone()

    mgr = CheckpointManager(str(tmp_path / "ckpts"))
    mgr.save(accelerator, step=500)

    with torch.no_grad():
        accelerator.unwrap_model(model).weight.fill_(0.0)
    assert not torch.equal(accelerator.unwrap_model(model).weight.data, original_weight)

    latest = mgr.find_latest()
    assert latest is not None
    accelerator.load_state(str(latest))

    assert torch.equal(accelerator.unwrap_model(model).weight.data, original_weight)
    assert mgr.load_step(latest) == 500

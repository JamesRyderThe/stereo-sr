from __future__ import annotations

import torch
import torch.nn as nn

from sissr.train.ema import ExponentialMovingAverage


def _make_model() -> nn.Module:
    return nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 8, 3, padding=1))


def test_shadow_independence() -> None:
    model = _make_model()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    original = [p.clone() for p in ema.shadow_params]
    nn.init.zeros_(model[1].weight)
    for orig, shadow in zip(original, ema.shadow_params, strict=True):
        assert torch.equal(orig, shadow)


def test_update_moves_toward_model() -> None:
    model = _make_model()
    nn.init.zeros_(model[1].weight)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    nn.init.ones_(model[1].weight)
    ema.update()
    assert ema.shadow_params[2].mean().item() > 0.0
    assert ema.shadow_params[2].mean().item() < 1.0


def test_warmup_decay() -> None:
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    nn.init.ones_(model.weight)
    ema.update()
    expected_decay = 2.0 / 11.0
    expected_val = 1.0 - expected_decay
    assert abs(ema.shadow_params[0].item() - expected_val) < 1e-5


def test_many_updates_converge() -> None:
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    nn.init.ones_(model.weight)
    for _ in range(500):
        ema.update()
    assert ema.shadow_params[0].item() > 0.99


def test_average_parameters_context() -> None:
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    model.weight.data.fill_(1.0)
    ema.update()

    model.weight.data.fill_(5.0)

    with ema.average_parameters():
        assert model.weight.data.item() < 1.0
    assert torch.equal(model.weight.data, torch.tensor([[5.0]]))


def test_state_dict_round_trip() -> None:
    model = _make_model()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.997)
    for _ in range(5):
        ema.update()
    state = ema.state_dict()

    ema2 = ExponentialMovingAverage(model.parameters(), decay=0.5)
    ema2.load_state_dict(state)
    assert ema2.num_updates == 5
    assert abs(ema2.decay - 0.997) < 1e-7
    for p1, p2 in zip(ema.shadow_params, ema2.shadow_params, strict=True):
        assert torch.equal(p1, p2)


def test_no_gradients_on_shadow() -> None:
    model = _make_model()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    for p in ema.shadow_params:
        assert not p.requires_grad

from __future__ import annotations

import pytest
import torch

from sissr.losses.charbonnier import charbonnier_loss


def test_identical_inputs_returns_eps() -> None:
    x = torch.randn(2, 3, 16, 16)
    eps = 1e-12
    loss = charbonnier_loss(x, x, eps=eps)
    assert loss.item() == pytest.approx(eps, rel=1e-4)


def test_known_hand_computed_value() -> None:
    pred = torch.tensor([3.0, 4.0])
    target = torch.zeros(2)
    eps = 1e-12
    expected = (
        torch.sqrt(torch.tensor(9.0 + eps**2)) + torch.sqrt(torch.tensor(16.0 + eps**2))
    ) / 2
    loss = charbonnier_loss(pred, target, eps=eps)
    assert torch.allclose(loss, expected)


def test_gradient_flows_to_both_inputs() -> None:
    pred = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randn(2, 3, 8, 8, requires_grad=True)
    loss = charbonnier_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert target.grad is not None
    assert pred.grad.abs().sum() > 0
    assert target.grad.abs().sum() > 0


def test_symmetry() -> None:
    a = torch.randn(2, 3, 8, 8)
    b = torch.randn(2, 3, 8, 8)
    assert charbonnier_loss(a, b).item() == charbonnier_loss(b, a).item()


def test_approaches_l1_for_large_errors() -> None:
    pred = torch.randn(4, 3, 16, 16) * 100.0 + 200.0
    target = torch.zeros(4, 3, 16, 16)
    eps = 1e-12
    charb = charbonnier_loss(pred, target, eps=eps)
    l1 = (pred - target).abs().mean()
    relative_error = (charb - l1).abs() / l1
    assert relative_error.item() < 0.01


def test_eps_zero_degenerates_to_exact_l1() -> None:
    pred = torch.randn(4, 3, 16, 16)
    target = torch.randn(4, 3, 16, 16)
    charb = charbonnier_loss(pred, target, eps=0.0)
    l1 = (pred - target).abs().mean()
    assert torch.allclose(charb, l1)

from __future__ import annotations

import copy
import weakref
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, parameters: Iterable[nn.Parameter], decay: float) -> None:
        params = list(parameters)
        self.decay = decay
        self.num_updates = 0
        self.shadow_params = [p.detach().clone() for p in params]
        self._params_refs = [weakref.ref(p) for p in params]
        self._collected_params: list[torch.Tensor] | None = None

    def _resolve_params(self) -> list[nn.Parameter]:
        resolved = [ref() for ref in self._params_refs]
        if any(p is None for p in resolved):
            raise ValueError("EMA parameters no longer exist")
        return [cast(nn.Parameter, p) for p in resolved]

    @torch.no_grad()
    def update(self) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for shadow, param in zip(self.shadow_params, self._resolve_params(), strict=True):
            shadow.lerp_(param.data, 1.0 - decay)

    def copy_to_model(self) -> None:
        for shadow, param in zip(self.shadow_params, self._resolve_params(), strict=True):
            param.data.copy_(shadow)

    @contextmanager
    def average_parameters(self) -> Generator[None, None, None]:
        params = self._resolve_params()
        self._collected_params = [p.data.clone() for p in params]
        self.copy_to_model()
        try:
            yield
        finally:
            for stored, param in zip(self._collected_params, params, strict=True):
                param.data.copy_(stored)
            self._collected_params = None

    def to(self, *, device: torch.device | str | None = None) -> None:
        self.shadow_params = [t.to(device=device) for t in self.shadow_params]

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        data = copy.deepcopy(state_dict)
        self.decay = float(data["decay"])
        self.num_updates = int(data["num_updates"])
        shadow = data["shadow_params"]
        if len(shadow) != len(self.shadow_params):
            raise ValueError(
                f"EMA param count mismatch: {len(shadow)} vs {len(self.shadow_params)}"
            )
        params = self._resolve_params()
        self.shadow_params = [
            t.to(device=p.device, dtype=p.dtype) for t, p in zip(shadow, params, strict=True)
        ]

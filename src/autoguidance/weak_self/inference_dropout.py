"""Construction — Inference-time dropout weak self.

Turn the model into a weaker predictor of the *same* conditional by enabling
dropout at inference time: every ``nn.Dropout`` module is switched to training
mode with probability ``p`` for the weak forward pass, then restored to its
original ``eval()`` state and original ``p`` afterwards. A fixed seed is set
before the forward pass so the stochastic mask is reproducible across calls.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from torch import LongTensor, FloatTensor

from autoguidance.weak_self.base import WeakSelf
from autoguidance.models.base import ModelAdapter


@contextmanager
def _dropout_on(adapter: ModelAdapter, p: float):
    """Enable every nn.Dropout at probability p, restoring training-mode + p on exit."""
    saved = []  # (module, was_training, old_p)
    n = 0
    for mod in adapter._model.modules():
        if isinstance(mod, nn.Dropout):
            saved.append((mod, mod.training, mod.p))
            mod.p = p
            mod.train()
            n += 1
    print(f"[InferenceDropout] enabled {n} nn.Dropout modules at p={p}")
    try:
        yield
    finally:
        for mod, was_training, old_p in saved:
            mod.p = old_p
            mod.train(was_training) if was_training else mod.eval()
        print(f"[InferenceDropout] restored {n} nn.Dropout modules (eval + original p)")


class InferenceDropoutWeakSelf(WeakSelf):
    """Weak self = the full model run once with dropout active (seeded)."""

    def __init__(self, p: float = 0.1, seed: int = 42) -> None:
        if not (0.0 < p < 1.0):
            raise ValueError("dropout p must be in (0, 1)")
        self.p = p
        self.seed = seed

    def __call__(
        self,
        x_t: LongTensor,
        cond: Optional[dict],
        adapter: ModelAdapter,
    ) -> FloatTensor:
        with _dropout_on(adapter, self.p):
            torch.manual_seed(self.seed)
            return adapter.logits(x_t, cond)

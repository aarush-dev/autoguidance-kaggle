"""Construction A — Input Noise weak self.

Two sub-variants:
  - InputNoiseRemask: re-mask p% of already-unmasked tokens before the weak forward pass.
  - InputNoiseGauss:  add Gaussian noise to input embeddings before the weak forward pass.

Both are one extra forward pass through the same model, matching the
in-situ autoguidance approach (arXiv:2510.17136).
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import LongTensor, FloatTensor

from autoguidance.weak_self.base import WeakSelf
from autoguidance.models.base import ModelAdapter


class InputNoiseRemask(WeakSelf):
    """Re-mask a fraction of already-unmasked tokens before the weak forward pass."""

    def __init__(self, remask_rate: float = 0.20, seed: int = 42) -> None:
        self.remask_rate = remask_rate
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def __call__(
        self,
        x_t: LongTensor,
        cond: Optional[dict],
        adapter: ModelAdapter,
    ) -> FloatTensor:
        mask_id = adapter.mask_token_id
        x_weak = x_t.clone()
        # Only re-mask positions that are NOT already masked
        unmasked = (x_t != mask_id)
        for b in range(x_t.shape[0]):
            pos = unmasked[b].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            n_to_remask = max(1, int(len(pos) * self.remask_rate))
            # Reproducible shuffle
            perm = torch.randperm(len(pos), generator=self._rng)
            chosen = pos[perm[:n_to_remask]]
            x_weak[b, chosen] = mask_id
        return adapter.logits(x_weak, cond)


class InputNoiseGauss(WeakSelf):
    """Add Gaussian noise to input embeddings before the weak forward pass."""

    def __init__(self, sigma: float = 0.1, seed: int = 42) -> None:
        self.sigma = sigma
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def __call__(
        self,
        x_t: LongTensor,
        cond: Optional[dict],
        adapter: ModelAdapter,
    ) -> FloatTensor:
        embeds = adapter.embed(x_t)  # [B, L, d_model]
        # self._rng is a CPU generator; embeds live on CUDA. Generating noise
        # directly on a CUDA tensor with a CPU generator raises. Draw on CPU
        # (fp32 — CPU normal_ has no half kernel), then cast to the embed device.
        noise = torch.empty(embeds.shape, dtype=torch.float32).normal_(generator=self._rng)
        noisy_embeds = embeds + (noise.to(embeds.device, embeds.dtype) * self.sigma)
        return adapter.logits_from_embed(noisy_embeds, cond)

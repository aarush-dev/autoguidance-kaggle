"""Construction B — Low-NFE weak self.

Use the model's single-step (or few-step) prediction of the clean sequence as
the weak logits. This is a research question: single-step predictions on
near-fully-masked inputs may be off-distribution (the model was barely trained
in that regime). Phase 0 checks whether this behaves as "same errors, amplified"
or as a qualitatively different model.
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import LongTensor, FloatTensor

from autoguidance.weak_self.base import WeakSelf
from autoguidance.models.base import ModelAdapter


class LowNFEWeakSelf(WeakSelf):
    """Run the model's forward pass on a near-fully-masked version of x_t.

    nfe=1: The model sees a sequence where all positions are masked, producing
    a single-step "guess" of the full clean sequence.

    nfe>1: Iteratively unmask the most confident tokens for nfe steps, then
    return the logits from the last step.
    """

    def __init__(self, nfe: int = 1) -> None:
        if nfe < 1:
            raise ValueError("nfe must be ≥ 1")
        self.nfe = nfe

    def __call__(
        self,
        x_t: LongTensor,
        cond: Optional[dict],
        adapter: ModelAdapter,
    ) -> FloatTensor:
        mask_id = adapter.mask_token_id
        B, L = x_t.shape

        if self.nfe == 1:
            # Fully-masked input: single-step prediction
            x_fully_masked = torch.full_like(x_t, mask_id)
            return adapter.logits(x_fully_masked, cond)

        # Multi-step: greedy unmasking for nfe-1 steps, return logits at step nfe
        x = torch.full_like(x_t, mask_id)
        for step in range(self.nfe):
            logits = adapter.logits(x, cond)          # [B, L, vocab]
            if step == self.nfe - 1:
                return logits                          # return last step's logits

            # Unmask most confident positions (greedy)
            is_masked = (x == mask_id)
            if not is_masked.any():
                return logits

            n_remaining = self.nfe - step
            probs = torch.softmax(logits.float(), dim=-1)
            confidence = probs.max(dim=-1).values      # [B, L]
            confidence[~is_masked] = -1.0

            n_masked = is_masked.sum(dim=-1)           # [B]
            sampled = logits.argmax(dim=-1)            # [B, L] greedy

            for b in range(B):
                n_to_unmask = max(1, (n_masked[b].item() + n_remaining - 1) // n_remaining)
                n_to_unmask = min(n_to_unmask, n_masked[b].item())
                if n_to_unmask == 0:
                    continue
                top_idx = confidence[b].topk(n_to_unmask).indices
                x[b, top_idx] = sampled[b, top_idx]

        return adapter.logits(x, cond)

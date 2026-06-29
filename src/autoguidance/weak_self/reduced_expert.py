"""Construction C — Reduced-expert routing weak self (MoE only).

Run the SAME MoE model but with the router restricted to fewer experts per token
(e.g. top-4 instead of top-8). The weak pass reuses the shared/dense path and
fewer experts, so it is much cheaper than a second full pass — the "nearly free"
weak self claim. See moe_patch.reduce_moe_topk for the patch + the renorm caveat.

Raises NotImplementedError for non-MoE adapters (run_phase0 catches + skips).
"""
from __future__ import annotations
from typing import Optional

from torch import LongTensor, FloatTensor

from autoguidance.weak_self.base import WeakSelf
from autoguidance.models.base import ModelAdapter
from autoguidance.weak_self.moe_patch import reduce_moe_topk, list_moe_modules


class ReducedExpertWeakSelf(WeakSelf):
    """Weak self = the full MoE model with router top-k reduced to ``top_k``."""

    def __init__(self, top_k: int = 4) -> None:
        if top_k < 1:
            raise ValueError("reduced_expert top_k must be ≥ 1")
        self.top_k = top_k

    def __call__(
        self,
        x_t: LongTensor,
        cond: Optional[dict],
        adapter: ModelAdapter,
    ) -> FloatTensor:
        if not getattr(adapter, "is_moe", False):
            raise NotImplementedError(
                "[ReducedExpert] adapter is not MoE (is_moe=False); "
                "reduced-expert routing weak self is unavailable for this model."
            )
        print(f"[ReducedExpert] reducing router to top_k={self.top_k}")
        list_moe_modules(adapter._model)
        with reduce_moe_topk(adapter._model, self.top_k):
            return adapter.logits(x_t, cond)

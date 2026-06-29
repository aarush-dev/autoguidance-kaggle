from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from torch import LongTensor, FloatTensor

from autoguidance.models.base import ModelAdapter


class WeakSelf(ABC):
    """Produces a deliberately weaker set of logits for the same conditional input.

    All implementations return logits in the same shape as adapter.logits():
    [batch, seq, vocab] float32.
    """

    @abstractmethod
    def __call__(
        self,
        x_t: LongTensor,
        cond: Optional[dict],
        adapter: ModelAdapter,
    ) -> FloatTensor:
        """Return weak logits [batch, seq, vocab].

        Args:
            x_t: Current token sequence (same as given to full model).
            cond: Same conditioning as the full model pass.
            adapter: The ModelAdapter for the full model.
        """

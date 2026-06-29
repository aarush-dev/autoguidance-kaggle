"""Construction — Layer-drop weak self.

Make a deliberately weaker forward pass by temporarily removing the last ``k``
decoder transformer layers. The remaining shallow stack is a lower-capacity
predictor of the same conditional, in the spirit of Karras autoguidance
(a weaker version of the *same* model).

Implementation: a context manager temporarily truncates the adapter's
``_decoder_layers()`` ModuleList to ``layers[:-k]`` (mutating ``_modules`` in
place so the model's ``forward`` — which iterates that same ModuleList object —
sees the shortened stack), then restores the full stack on exit. Nothing is
copied; the same layer modules are re-attached.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Optional

from torch import LongTensor, FloatTensor

from autoguidance.weak_self.base import WeakSelf
from autoguidance.models.base import ModelAdapter


@contextmanager
def _dropped_layers(adapter: ModelAdapter, k: int):
    """Temporarily drop the last ``k`` decoder layers, restoring on exit."""
    layers = adapter._decoder_layers()           # nn.ModuleList
    saved = list(layers._modules.items())        # [(key, module), ...]
    n = len(saved)
    if k >= n:
        raise NotImplementedError(
            f"[LayerDrop] cannot drop k={k} of only {n} layers (would leave none)."
        )
    kept = saved[: n - k]
    print(f"[LayerDrop] dropping last {k} of {n} layers → keeping {len(kept)} layers")
    try:
        layers._modules.clear()
        for key, mod in kept:
            layers._modules[key] = mod
        yield
    finally:
        layers._modules.clear()
        for key, mod in saved:
            layers._modules[key] = mod
        print(f"[LayerDrop] restored full stack of {len(saved)} layers")


class LayerDropWeakSelf(WeakSelf):
    """Weak self = full model with the last ``k`` decoder layers removed."""

    def __init__(self, k: int = 2) -> None:
        if k < 1:
            raise ValueError("layer_drop k must be ≥ 1")
        self.k = k

    def unavailable_reason(self, adapter: ModelAdapter) -> Optional[str]:
        n = adapter.n_layers
        if self.k >= n:
            return f"cannot drop k={self.k} of only {n} layers (would leave none)"
        return None

    def __call__(
        self,
        x_t: LongTensor,
        cond: Optional[dict],
        adapter: ModelAdapter,
    ) -> FloatTensor:
        with _dropped_layers(adapter, self.k):
            return adapter.logits(x_t, cond)

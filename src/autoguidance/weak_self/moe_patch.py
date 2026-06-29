"""MoE router patching utilities for the reduced-expert weak self.

``reduce_moe_topk(model, new_top_k)`` is a context manager that temporarily
lowers the number of experts routed per token for every MoE router submodule it
can find, restoring the original values on exit. ``list_moe_modules(model)``
prints what it would patch (debug aid for confirming the router layout at build
time).

RENORM CAVEAT
-------------
Lowering top_k is only a *clean* weak self if the router re-softmaxes the gate
weights over the selected-k experts (i.e. the kept experts' weights are
renormalised to sum to 1). If the router instead applies a fixed softmax over
ALL experts and then simply gathers the top-k, dropping experts removes weight
mass and rescales the block output — that is a different perturbation than
"same model, fewer experts", and the Phase 0 sanity checks may flag it.

Most modern HF MoE implementations (Mixtral/Qwen-MoE/Gemma-MoE style) renormalise
over the selected-k, so reducing top_k is clean. If the sanity check fails, a
forward-wrap fallback can be added here: wrap the router's forward to re-softmax
the gate logits over the kept-k experts before weighting. That is intentionally
NOT implemented yet (YAGNI) — add it only if the clean path fails its check.
"""
from __future__ import annotations
from contextlib import contextmanager

import torch.nn as nn

# Common attribute names that hold the per-token expert count across HF MoE impls.
_TOPK_ATTRS = ("top_k", "top_k_experts", "num_experts_per_tok")


def _matched_modules(model: nn.Module):
    """Yield (module, attr, value) for every submodule holding an int top-k > 1."""
    for mod in model.modules():
        for attr in _TOPK_ATTRS:
            val = getattr(mod, attr, None)
            if isinstance(val, int) and not isinstance(val, bool) and val > 1:
                yield mod, attr, val


def list_moe_modules(model: nn.Module) -> None:
    """PRINT every router submodule + matched top-k attr/value (debug)."""
    found = 0
    for mod, attr, val in _matched_modules(model):
        print(f"[moe_patch] {type(mod).__name__}.{attr} = {val}")
        found += 1
    if found == 0:
        print("[moe_patch] no MoE top-k attributes found "
              f"(searched {_TOPK_ATTRS}); model may not be MoE or uses other names.")
    else:
        print(f"[moe_patch] {found} MoE router attribute(s) matched.")


@contextmanager
def reduce_moe_topk(model: nn.Module, new_top_k: int):
    """Temporarily set every router top-k attr to ``new_top_k``; restore on exit."""
    saved = []  # (module, attr, old_value)
    for mod, attr, val in _matched_modules(model):
        if new_top_k >= val:
            print(f"[moe_patch] skip {type(mod).__name__}.{attr}={val} "
                  f"(new_top_k={new_top_k} not smaller)")
            continue
        saved.append((mod, attr, val))
        setattr(mod, attr, new_top_k)
        print(f"[moe_patch] {type(mod).__name__}.{attr}: {val} → {new_top_k}")
    if not saved:
        print(f"[moe_patch] WARNING: nothing patched to top_k={new_top_k}; "
              "weak pass will equal the full pass.")
    try:
        yield
    finally:
        for mod, attr, old in saved:
            setattr(mod, attr, old)
        print(f"[moe_patch] restored {len(saved)} router attribute(s).")

"""Per-position Shannon entropy for Phase 0 Check 1.

Check 1: weak self is more uncertain (higher entropy) than full model,
not just wrong in new places.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import FloatTensor


def _entropy(logits: FloatTensor) -> np.ndarray:
    """Shannon entropy in nats. Input: [..., vocab]. Returns [...] float32 numpy."""
    return entropy_nats(logits).cpu().numpy()


def entropy_nats(logits: FloatTensor) -> FloatTensor:
    """Shannon entropy in nats, kept on the input device. Input: [..., vocab] → [...]."""
    probs = torch.softmax(logits.float(), dim=-1)
    # Clamp for numerical stability
    probs = probs.clamp(min=1e-9)
    return -(probs * probs.log()).sum(dim=-1)


def entropy_stats_from_arrays(
    flat_full: np.ndarray,
    flat_weak: np.ndarray,
    check1_entropy_fraction: float = 0.80,
) -> dict:
    """Check 1 from pre-reduced per-masked-position entropy arrays (streaming path)."""
    delta = flat_weak - flat_full
    mean_full = float(flat_full.mean())
    mean_weak = float(flat_weak.mean())
    fraction_higher = float((delta > 0).mean())
    check1_pass = (mean_weak > mean_full) and (fraction_higher >= check1_entropy_fraction)
    return {
        "full_entropy": flat_full,
        "weak_entropy": flat_weak,
        "delta_entropy": delta,
        "mean_full": mean_full,
        "mean_weak": mean_weak,
        "fraction_weak_higher": fraction_higher,
        "check1_pass": check1_pass,
    }


def compute_entropy_stats(
    full_logits: FloatTensor,
    weak_logits: FloatTensor,
    mask_positions: np.ndarray,
    check1_entropy_fraction: float = 0.80,
) -> dict:
    """Compute per-position entropy and Check 1 pass/fail.

    Args:
        full_logits: [B, L, vocab] full model logits.
        weak_logits: [B, L, vocab] weak model logits.
        mask_positions: [B, L] bool numpy array — True at positions to evaluate.
        check1_entropy_fraction: fraction of positions that must have H_weak > H_full.

    Returns dict with:
        full_entropy: flat numpy array of entropies at masked positions
        weak_entropy: flat numpy array of entropies at masked positions
        delta_entropy: weak - full
        mean_full: float
        mean_weak: float
        fraction_weak_higher: float (0-1)
        check1_pass: bool
    """
    H_full = _entropy(full_logits)     # [B, L]
    H_weak = _entropy(weak_logits)     # [B, L]

    # Flatten to masked positions only
    return entropy_stats_from_arrays(
        H_full[mask_positions], H_weak[mask_positions], check1_entropy_fraction
    )

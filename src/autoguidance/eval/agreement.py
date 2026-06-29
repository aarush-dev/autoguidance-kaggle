"""Top-1 agreement and Spearman rank correlation for Phase 0 Check 2.

Check 2: the weak self's argmax still aligns with the full model's most of the time,
and the logit rankings are correlated (same direction, not a different task).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
import torch
from torch import FloatTensor


def compute_agreement_stats(
    full_logits: FloatTensor,
    weak_logits: FloatTensor,
    mask_positions: np.ndarray,
    check2_top1_agreement: float = 0.60,
    check2_spearman_rho: float = 0.50,
) -> dict:
    """Compute top-1 agreement rate and mean Spearman rank correlation.

    Args:
        full_logits: [B, L, vocab]
        weak_logits: [B, L, vocab]
        mask_positions: [B, L] bool — True at positions to evaluate.
        check2_top1_agreement: minimum agreement rate to pass.
        check2_spearman_rho: minimum mean Spearman ρ to pass.

    Returns dict with:
        top1_agreement: float
        spearman_rho_mean: float
        spearman_rho_std: float
        check2_pass: bool
    """
    # Argmax for each position
    full_top1 = full_logits.argmax(dim=-1).cpu().numpy()   # [B, L]
    weak_top1 = weak_logits.argmax(dim=-1).cpu().numpy()   # [B, L]

    full_flat = full_top1[mask_positions]
    weak_flat = weak_top1[mask_positions]

    # Spearman rank correlation per position (sample up to 2000 positions for speed)
    full_np = full_logits.cpu().float().numpy()
    weak_np = weak_logits.cpu().float().numpy()

    rhos = []
    positions = list(zip(*np.where(mask_positions)))
    if len(positions) > 2000:
        idx = np.random.default_rng(seed=0).choice(len(positions), 2000, replace=False)
        positions = [positions[i] for i in idx]

    for b, l in positions:
        rho, _ = spearmanr(full_np[b, l], weak_np[b, l])
        if not np.isnan(rho):
            rhos.append(rho)

    return agreement_stats_from_arrays(
        full_flat, weak_flat, rhos, check2_top1_agreement, check2_spearman_rho
    )


def spearman_rho(full_logit_vec, weak_logit_vec):
    """Spearman ρ between two per-position logit vectors. NaN-safe → None if degenerate."""
    rho, _ = spearmanr(full_logit_vec, weak_logit_vec)
    return None if np.isnan(rho) else float(rho)


def agreement_stats_from_arrays(
    full_top1: np.ndarray,
    weak_top1: np.ndarray,
    rhos: list,
    check2_top1_agreement: float = 0.60,
    check2_spearman_rho: float = 0.50,
) -> dict:
    """Check 2 from pre-reduced per-masked-position top-1 arrays + precomputed rhos."""
    top1_agreement = float((full_top1 == weak_top1).mean())
    spearman_mean = float(np.mean(rhos)) if rhos else 0.0
    spearman_std = float(np.std(rhos)) if rhos else 0.0
    check2_pass = (
        top1_agreement >= check2_top1_agreement
        and spearman_mean >= check2_spearman_rho
    )
    return {
        "top1_agreement": top1_agreement,
        "spearman_rho_mean": spearman_mean,
        "spearman_rho_std": spearman_std,
        "check2_pass": check2_pass,
    }

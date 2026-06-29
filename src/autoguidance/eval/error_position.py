"""Error-position correlation for Phase 0 Check 3.

Check 3: the positions where full and weak models disagree most are the same
positions where the full model makes errors against ground truth.
If true, the weak self exaggerates the full model's real error sites —
the necessary precondition for autoguidance to work.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr
import torch
from torch import FloatTensor, LongTensor


def compute_error_position_stats(
    full_logits: FloatTensor,
    weak_logits: FloatTensor,
    ground_truth: LongTensor,
    mask_positions: np.ndarray,
    check3_pearson_r: float = 0.20,
    check3_precision: float = 0.50,
) -> dict:
    """Check 3: disagreement concentrates at full-model error sites.

    Args:
        full_logits: [B, L, vocab]
        weak_logits: [B, L, vocab]
        ground_truth: [B, L] true token IDs at all positions.
        mask_positions: [B, L] bool — True at originally-masked positions to evaluate.
        check3_pearson_r: minimum Pearson r to pass via correlation criterion.
        check3_precision: minimum precision to pass via prec/recall criterion.

    Returns dict with:
        pearson_r: float
        pearson_p: float
        precision: float  (disagreement→ predicts full model error)
        recall: float
        full_error_rate: float
        disagreement_rate: float
        check3_pass: bool
    """
    full_top1 = full_logits.argmax(dim=-1).cpu().numpy()   # [B, L]
    weak_top1 = weak_logits.argmax(dim=-1).cpu().numpy()   # [B, L]
    gt_np = ground_truth.cpu().numpy()                      # [B, L]

    # Max logit difference magnitude at each position (as disagreement signal)
    diff_magnitude = (full_logits - weak_logits).abs().max(dim=-1).values.cpu().numpy()  # [B, L]

    # Restrict to masked positions
    return error_position_stats_from_arrays(
        diff_magnitude[mask_positions],
        full_top1[mask_positions],
        weak_top1[mask_positions],
        gt_np[mask_positions],
        check3_pearson_r,
        check3_precision,
    )


def error_position_stats_from_arrays(
    flat_diff_mag: np.ndarray,
    flat_full_top1: np.ndarray,
    flat_weak_top1: np.ndarray,
    flat_gt: np.ndarray,
    check3_pearson_r: float = 0.20,
    check3_precision: float = 0.50,
) -> dict:
    """Check 3 from pre-reduced per-masked-position arrays (streaming path)."""
    # Full model error indicator: argmax(full) ≠ ground_truth
    full_error = (flat_full_top1 != flat_gt).astype(np.float32)
    # Disagreement indicator: argmax(weak) ≠ argmax(full)
    disagreement = (flat_weak_top1 != flat_full_top1).astype(np.float32)

    # Pearson correlation between disagreement magnitude and full-model error
    if len(flat_diff_mag) < 3:
        r, p = 0.0, 1.0
    else:
        r, p = pearsonr(flat_diff_mag, full_error)
        if np.isnan(r):
            r, p = 0.0, 1.0

    # Precision / recall: treat disagreement as a binary classifier for full-model errors
    tp = float((disagreement * full_error).sum())
    fp = float((disagreement * (1 - full_error)).sum())
    fn = float(((1 - disagreement) * full_error).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    full_error_rate = float(full_error.mean())
    disagreement_rate = float(disagreement.mean())

    check3_pass = (r >= check3_pearson_r) or (precision >= check3_precision)

    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "precision": precision,
        "recall": recall,
        "full_error_rate": full_error_rate,
        "disagreement_rate": disagreement_rate,
        "check3_pass": check3_pass,
    }

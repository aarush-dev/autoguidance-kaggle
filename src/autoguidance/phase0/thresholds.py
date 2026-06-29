"""Phase 0: threshold evaluation (model-free).

Reloads the reduced arrays dumped by characterize.py and applies the three
pass/fail criteria. Pure numpy — no torch model is loaded here, so thresholds can
be re-evaluated or swept cheaply without re-running the expensive forward passes.

    check1 — entropy: weak self is more uncertain than the full model
    check2 — agreement: top-1 + Spearman rank correlation (same direction)
    check3 — error position: disagreement concentrates at full-model error sites
"""
from __future__ import annotations
from typing import Dict

import numpy as np

from autoguidance.eval.entropy import entropy_stats_from_arrays
from autoguidance.eval.agreement import agreement_stats_from_arrays
from autoguidance.eval.error_position import error_position_stats_from_arrays


def evaluate_arrays(arrays: Dict, cfg) -> Dict:
    """Apply check1/check2/check3 to one construction's reduced arrays.

    Args:
        arrays: dict from arrays.load_arrays (entropy_full, entropy_weak, top1_full,
            top1_weak, diff_mag, gt, rho_sample, meta).
        cfg: Phase0Config carrying the threshold knobs.

    Returns:
        dict with check1/check2/check3 sub-dicts plus all_pass + bookkeeping.
    """
    ent_full = np.asarray(arrays["entropy_full"])
    ent_weak = np.asarray(arrays["entropy_weak"])
    top1_full = np.asarray(arrays["top1_full"])
    top1_weak = np.asarray(arrays["top1_weak"])
    diff_mag = np.asarray(arrays["diff_mag"])
    gt = np.asarray(arrays["gt"])
    rho_sample = np.asarray(arrays.get("rho_sample", np.array([], dtype=np.float32)))
    rhos = [float(x) for x in rho_sample.ravel().tolist()]

    name = arrays.get("meta", {}).get("construction", "?") if isinstance(arrays.get("meta"), dict) else "?"

    check1 = entropy_stats_from_arrays(
        ent_full, ent_weak, check1_entropy_fraction=cfg.check1_entropy_fraction,
    )
    check2 = agreement_stats_from_arrays(
        top1_full, top1_weak, rhos,
        check2_top1_agreement=cfg.check2_top1_agreement,
        check2_spearman_rho=cfg.check2_spearman_rho,
    )
    check3 = error_position_stats_from_arrays(
        diff_mag, top1_full, top1_weak, gt,
        check3_pearson_r=cfg.check3_pearson_r,
        check3_precision=cfg.check3_precision,
    )

    all_pass = bool(check1["check1_pass"] and check2["check2_pass"] and check3["check3_pass"])

    print(
        f"  [{name}] "
        f"C1={'PASS' if check1['check1_pass'] else 'FAIL'} "
        f"(H_full={check1['mean_full']:.3f} H_weak={check1['mean_weak']:.3f} "
        f"frac_higher={check1['fraction_weak_higher']:.3f}) | "
        f"C2={'PASS' if check2['check2_pass'] else 'FAIL'} "
        f"(top1={check2['top1_agreement']:.3f} rho={check2['spearman_rho_mean']:.3f}) | "
        f"C3={'PASS' if check3['check3_pass'] else 'FAIL'} "
        f"(r={check3['pearson_r']:.3f} prec={check3['precision']:.3f}) "
        f"=> {'ALL PASS' if all_pass else 'FAIL'}"
    )

    return {
        "check1": check1,
        "check2": check2,
        "check3": check3,
        "all_pass": all_pass,
        "n_masked_positions": int(ent_full.shape[0]),
    }


def evaluate_dir(arrays_dir: str, cfg) -> Dict[str, Dict]:
    """Evaluate every construction npz found at the top level of arrays_dir."""
    from autoguidance.phase0.arrays import load_arrays, list_constructions

    names = list_constructions(arrays_dir)
    print(f"\n[Phase 0] evaluate_dir: {arrays_dir} -> {names}")
    verdicts: Dict[str, Dict] = {}
    for name in names:
        try:
            arrays = load_arrays(arrays_dir, name)
            verdicts[name] = evaluate_arrays(arrays, cfg)
        except Exception as e:
            print(f"  [{name}] ERROR evaluating: {e}")
            verdicts[name] = {"error": str(e)}
    passing = [n for n, v in verdicts.items() if v.get("all_pass", False)]
    print(f"[Phase 0] evaluate_dir done — passing: {passing if passing else 'NONE'}")
    return verdicts

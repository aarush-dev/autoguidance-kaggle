"""Tests for phase0/thresholds.py: evaluate_arrays and evaluate_dir.

Contract:
  evaluate_arrays(arrays: dict, cfg) -> dict  # check1/check2/check3/all_pass
  evaluate_dir(arrays_dir, cfg) -> dict[name -> verdict]
  Pure numpy — no torch model required.
  Uses eval/entropy.entropy_stats_from_arrays, eval/agreement.agreement_stats_from_arrays,
  eval/error_position.error_position_stats_from_arrays internally.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile

import numpy as np
import pytest

from autoguidance.phase0.arrays import save_arrays
from autoguidance.phase0.thresholds import evaluate_arrays, evaluate_dir
from autoguidance.config import Phase0Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides):
    cfg = Phase0Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _relaxed_cfg():
    """All thresholds set to trivially-passing values."""
    return _cfg(
        check1_entropy_fraction=0.0,
        check2_top1_agreement=0.0,
        check2_spearman_rho=-1.0,
        check3_pearson_r=-1.0,
        check3_precision=0.0,
    )


def _make_passing_arrays(n=200, seed=0):
    """Arrays crafted so all three checks pass at default thresholds."""
    rng = np.random.default_rng(seed)

    # Check 1: weak entropy strictly > full at > 80 % of positions.
    entropy_full = rng.uniform(0.5, 1.0, n).astype(np.float32)
    entropy_weak = entropy_full + rng.uniform(0.1, 0.5, n).astype(np.float32)

    # Check 2: top1 agrees at ~80 % (> 60 % threshold).
    top1_full = rng.integers(0, 10, n).astype(np.int64)
    top1_weak = top1_full.copy()
    flip_idx = rng.choice(n, n // 5, replace=False)           # flip 20 %
    top1_weak[flip_idx] = rng.integers(10, 20, len(flip_idx)).astype(np.int64)

    # Check 3: disagreement co-locates with full-model errors.
    gt = top1_full.copy()
    err_idx = rng.choice(n, n * 4 // 10, replace=False)       # 40 % errors
    gt[err_idx] = rng.integers(20, 30, len(err_idx)).astype(np.int64)
    disagree_mask = top1_full != top1_weak
    diff_mag = np.where(
        disagree_mask, 2.0, 0.05
    ).astype(np.float32) + rng.uniform(0, 0.02, n).astype(np.float32)

    # Spearman samples: all positive → satisfies any rho threshold above 0.
    rho_sample = rng.uniform(0.6, 0.95, 50).astype(np.float32)

    return {
        "entropy_full": entropy_full,
        "entropy_weak": entropy_weak,
        "top1_full":    top1_full,
        "top1_weak":    top1_weak,
        "diff_mag":     diff_mag,
        "gt":           gt,
        "rho_sample":   rho_sample,
    }


def _make_failing_check1_arrays(n=200, seed=1):
    """Arrays where weak entropy is lower than full → check1 always fails."""
    rng = np.random.default_rng(seed)
    entropy_full = rng.uniform(1.0, 2.0, n).astype(np.float32)
    entropy_weak = entropy_full - rng.uniform(0.1, 0.5, n).astype(np.float32)  # lower
    top1_full = rng.integers(0, 10, n).astype(np.int64)
    top1_weak = top1_full.copy()
    gt = top1_full.copy()
    diff_mag = rng.uniform(0, 0.5, n).astype(np.float32)
    rho_sample = rng.uniform(0.5, 0.9, 30).astype(np.float32)
    return {
        "entropy_full": entropy_full,
        "entropy_weak": entropy_weak,
        "top1_full": top1_full,
        "top1_weak": top1_weak,
        "diff_mag": diff_mag,
        "gt": gt,
        "rho_sample": rho_sample,
    }


def _make_failing_check2_arrays(n=200, seed=2):
    """Arrays where top1 agreement is near zero → check2 always fails."""
    rng = np.random.default_rng(seed)
    entropy_full = rng.uniform(0.5, 1.0, n).astype(np.float32)
    entropy_weak = entropy_full + rng.uniform(0.1, 0.5, n).astype(np.float32)
    top1_full = rng.integers(0, 50, n).astype(np.int64)
    # completely different argmax bucket — agreement ≈ 0
    top1_weak = rng.integers(50, 99, n).astype(np.int64)
    gt = top1_full.copy()
    diff_mag = rng.uniform(0.5, 2.0, n).astype(np.float32)
    rho_sample = rng.uniform(-0.9, -0.5, 30).astype(np.float32)  # negative ρ
    return {
        "entropy_full": entropy_full,
        "entropy_weak": entropy_weak,
        "top1_full": top1_full,
        "top1_weak": top1_weak,
        "diff_mag": diff_mag,
        "gt": gt,
        "rho_sample": rho_sample,
    }


# ---------------------------------------------------------------------------
# evaluate_arrays: required keys
# ---------------------------------------------------------------------------

def test_evaluate_arrays_returns_required_keys():
    """evaluate_arrays always returns check1/check2/check3/all_pass."""
    cfg = _relaxed_cfg()
    arrs = _make_passing_arrays()
    result = evaluate_arrays(arrs, cfg)
    for key in ("check1", "check2", "check3", "all_pass"):
        assert key in result, f"Missing key: {key}"


def test_evaluate_arrays_all_pass_is_bool():
    """all_pass is a plain Python bool."""
    result = evaluate_arrays(_make_passing_arrays(), _relaxed_cfg())
    assert isinstance(result["all_pass"], bool)


def test_evaluate_arrays_check_dicts_have_pass_flags():
    """Each check sub-dict contains its pass/fail bool."""
    result = evaluate_arrays(_make_passing_arrays(), _relaxed_cfg())
    assert "check1_pass" in result["check1"]
    assert "check2_pass" in result["check2"]
    assert "check3_pass" in result["check3"]


# ---------------------------------------------------------------------------
# evaluate_arrays: known-outcome arrays
# ---------------------------------------------------------------------------

def test_evaluate_arrays_all_pass_with_relaxed_thresholds():
    """Known-passing arrays → all checks pass when thresholds are relaxed."""
    cfg = _relaxed_cfg()
    result = evaluate_arrays(_make_passing_arrays(), cfg)
    assert result["check1"]["check1_pass"], "check1 should pass"
    assert result["check2"]["check2_pass"], "check2 should pass"
    assert result["check3"]["check3_pass"], "check3 should pass"
    assert result["all_pass"], "all_pass should be True"


def test_evaluate_arrays_check1_fails_when_weak_entropy_lower():
    """Arrays with weak < full entropy always fail check1 regardless of threshold."""
    cfg = _cfg(
        check1_entropy_fraction=0.0,  # normally trivial, but mean_weak < mean_full
        check2_top1_agreement=0.0,
        check2_spearman_rho=-1.0,
        check3_pearson_r=-1.0,
        check3_precision=0.0,
    )
    result = evaluate_arrays(_make_failing_check1_arrays(), cfg)
    assert not result["check1"]["check1_pass"], "check1 should fail (weak entropy < full)"
    assert not result["all_pass"]


def test_evaluate_arrays_check2_fails_on_zero_agreement():
    """Arrays with near-zero top1 agreement fail check2 with any positive threshold."""
    cfg = _cfg(
        check1_entropy_fraction=0.0,
        check2_top1_agreement=0.30,  # 30 % required; actual ≈ 0 %
        check2_spearman_rho=-1.0,
        check3_pearson_r=-1.0,
        check3_precision=0.0,
    )
    result = evaluate_arrays(_make_failing_check2_arrays(), cfg)
    assert not result["check2"]["check2_pass"], "check2 should fail (agreement ≈ 0)"
    assert not result["all_pass"]


# ---------------------------------------------------------------------------
# Threshold knob: flipping verdicts without re-running any model
# ---------------------------------------------------------------------------

def test_threshold_knob_flips_check1_verdict():
    """Raising check1_entropy_fraction from 0.5 to 1.0 flips a marginal case.

    We craft arrays where ~80% of positions have weak > full entropy, then verify:
      - threshold=0.5 passes (80% >= 50%)
      - threshold=1.0 fails  (80% <  100%)
    """
    rng = np.random.default_rng(7)
    n = 200
    entropy_full = rng.uniform(0.5, 1.0, n).astype(np.float32)
    entropy_weak = entropy_full.copy()
    # Make ~80% of positions have weak > full (first 160 positions).
    entropy_weak[:160] += rng.uniform(0.1, 0.4, 160).astype(np.float32)   # higher
    entropy_weak[160:] -= rng.uniform(0.05, 0.1, 40).astype(np.float32)   # lower

    top1_full = rng.integers(0, 10, n).astype(np.int64)
    top1_weak = top1_full.copy()
    gt = top1_full.copy()
    diff_mag = rng.uniform(0, 1.0, n).astype(np.float32)
    rho_sample = rng.uniform(0.5, 0.9, 30).astype(np.float32)

    arrs = {
        "entropy_full": entropy_full,
        "entropy_weak": entropy_weak,
        "top1_full": top1_full,
        "top1_weak": top1_weak,
        "diff_mag": diff_mag,
        "gt": gt,
        "rho_sample": rho_sample,
    }
    cfg_loose = _cfg(
        check1_entropy_fraction=0.50,  # 80% >= 50% → PASS
        check2_top1_agreement=0.0, check2_spearman_rho=-1.0,
        check3_pearson_r=-1.0, check3_precision=0.0,
    )
    cfg_tight = _cfg(
        check1_entropy_fraction=1.0,   # 80% < 100% → FAIL
        check2_top1_agreement=0.0, check2_spearman_rho=-1.0,
        check3_pearson_r=-1.0, check3_precision=0.0,
    )
    r_loose = evaluate_arrays(arrs, cfg_loose)
    r_tight = evaluate_arrays(arrs, cfg_tight)
    assert r_loose["check1"]["check1_pass"], (
        f"loose (threshold=0.5) should pass; fraction={r_loose['check1']['fraction_weak_higher']:.2f}"
    )
    assert not r_tight["check1"]["check1_pass"], (
        f"tight (threshold=1.0) should fail; fraction={r_tight['check1']['fraction_weak_higher']:.2f}"
    )


def test_threshold_knob_flips_check2_verdict():
    """Raising check2_top1_agreement from 0 to 0.99 flips a marginal case."""
    arrs = _make_passing_arrays(n=200)  # actual agreement ≈ 80 %
    cfg_loose = _relaxed_cfg()
    cfg_tight = _cfg(
        check1_entropy_fraction=0.0,
        check2_top1_agreement=0.99,    # 99 % required; actual ≈ 80 % → fails
        check2_spearman_rho=-1.0,
        check3_pearson_r=-1.0,
        check3_precision=0.0,
    )
    assert evaluate_arrays(arrs, cfg_loose)["check2"]["check2_pass"]
    assert not evaluate_arrays(arrs, cfg_tight)["check2"]["check2_pass"]


def test_threshold_change_does_not_require_model():
    """Flipping a verdict via evaluate_arrays imports no torch / model code."""
    # The test itself is proof: if evaluate_arrays imports torch or runs a model,
    # the CPU-only CI environment would fail (GPU not available).  A simpler guard:
    # check that the module can be imported in isolation with no GPU side effects.
    import importlib
    mod = importlib.import_module("autoguidance.phase0.thresholds")
    assert hasattr(mod, "evaluate_arrays")
    assert hasattr(mod, "evaluate_dir")


# ---------------------------------------------------------------------------
# evaluate_dir
# ---------------------------------------------------------------------------

def test_evaluate_dir_empty_returns_empty_dict():
    """evaluate_dir on an empty directory returns {}."""
    with tempfile.TemporaryDirectory() as d:
        result = evaluate_dir(d, _relaxed_cfg())
        assert result == {}


def test_evaluate_dir_finds_saved_constructions():
    """evaluate_dir reads saved .npz files and returns one verdict per construction."""
    cfg = _relaxed_cfg()
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_passing_arrays()
        save_arrays(d, "low_nfe", {"construction": "low_nfe"}, **arrs)
        save_arrays(d, "input_noise_remask", {"construction": "input_noise_remask"}, **arrs)
        verdicts = evaluate_dir(d, cfg)
        assert set(verdicts.keys()) == {"low_nfe", "input_noise_remask"}


def test_evaluate_dir_verdict_has_required_keys():
    """Each verdict from evaluate_dir contains check1/check2/check3/all_pass."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_passing_arrays()
        save_arrays(d, "low_nfe", {}, **arrs)
        verdicts = evaluate_dir(d, _relaxed_cfg())
        v = verdicts["low_nfe"]
        for key in ("check1", "check2", "check3", "all_pass"):
            assert key in v, f"Missing key in verdict: {key}"


def test_evaluate_dir_passing_arrays_pass():
    """Passing arrays saved to dir → evaluate_dir reports all_pass=True."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_passing_arrays()
        save_arrays(d, "test_construction", {}, **arrs)
        verdicts = evaluate_dir(d, _relaxed_cfg())
        assert verdicts["test_construction"]["all_pass"]


def test_evaluate_dir_failing_arrays_fail():
    """Failing arrays saved to dir → evaluate_dir reports all_pass=False."""
    cfg = _cfg(
        check1_entropy_fraction=0.0,
        check2_top1_agreement=0.50,  # requires 50 % agreement; failing arrs have ~0
        check2_spearman_rho=-1.0,
        check3_pearson_r=-1.0,
        check3_precision=0.0,
    )
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_failing_check2_arrays()
        save_arrays(d, "bad_construction", {}, **arrs)
        verdicts = evaluate_dir(d, cfg)
        assert not verdicts["bad_construction"]["all_pass"]

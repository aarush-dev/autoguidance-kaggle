"""Audit-regression tests: the two failure modes that produced the bogus
results.zip (mixed-model aggregation + degenerate-entropy silent FAIL)."""
import os
import sys
import types
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from autoguidance.phase0.arrays import save_arrays
from autoguidance.phase0.thresholds import evaluate_dir, evaluate_arrays


def _cfg():
    return types.SimpleNamespace(
        check1_entropy_fraction=0.80,
        check2_top1_agreement=0.60,
        check2_spearman_rho=0.50,
        check3_pearson_r=0.20,
        check3_precision=0.50,
    )


def _dump(arrays_dir, name, model, n=64, entropy_scale=2.0):
    rng = np.random.default_rng(0)
    ef = np.abs(rng.normal(entropy_scale, 0.1, n)).astype(np.float32)
    ew = ef + 0.1
    top1 = rng.integers(0, 100, n).astype(np.int64)
    save_arrays(
        arrays_dir, name, {"construction": name, "model": model},
        entropy_full=ef, entropy_weak=ew,
        top1_full=top1, top1_weak=top1.copy(),
        diff_mag=np.abs(rng.normal(1, 0.1, n)).astype(np.float32),
        gt=top1.copy(), rho_sample=np.full(10, 0.9, np.float32),
    )


def test_evaluate_dir_rejects_mixed_models():
    with tempfile.TemporaryDirectory() as d:
        _dump(d, "layer_drop", "LLaDAAdapter")
        _dump(d, "reduced_expert", "DiffusionGemmaAdapter")
        with pytest.raises(ValueError, match="mixes 2 models"):
            evaluate_dir(d, _cfg())


def test_evaluate_dir_allows_single_model():
    with tempfile.TemporaryDirectory() as d:
        _dump(d, "layer_drop", "LLaDAAdapter")
        _dump(d, "inference_dropout", "LLaDAAdapter")
        verdicts = evaluate_dir(d, _cfg())  # must not raise
        assert set(verdicts) == {"layer_drop", "inference_dropout"}


def test_degenerate_entropy_flagged_and_blocks_pass():
    """Near-zero full entropy (DiffusionGemma canvas-crop symptom) → flagged,
    check1 forced FAIL regardless of the fraction test."""
    n = 64
    ef = np.full(n, 1e-5, np.float32)          # ~deterministic full model
    ew = ef + 1e-6
    arrays = {
        "entropy_full": ef, "entropy_weak": ew,
        "top1_full": np.zeros(n, np.int64), "top1_weak": np.zeros(n, np.int64),
        "diff_mag": np.ones(n, np.float32), "gt": np.zeros(n, np.int64),
        "rho_sample": np.full(10, 0.9, np.float32),
        "meta": {"construction": "reduced_expert", "model": "DiffusionGemmaAdapter"},
    }
    r = evaluate_arrays(arrays, _cfg())
    assert r["check1"]["degenerate_full_entropy"] is True
    assert r["check1"]["check1_pass"] is False
    assert r["all_pass"] is False


def test_healthy_entropy_not_flagged():
    n = 64
    ef = np.full(n, 1.5, np.float32)
    ew = ef + 0.2
    arrays = {
        "entropy_full": ef, "entropy_weak": ew,
        "top1_full": np.zeros(n, np.int64), "top1_weak": np.zeros(n, np.int64),
        "diff_mag": np.ones(n, np.float32), "gt": np.zeros(n, np.int64),
        "rho_sample": np.full(10, 0.9, np.float32),
        "meta": {"construction": "layer_drop", "model": "LLaDAAdapter"},
    }
    r = evaluate_arrays(arrays, _cfg())
    assert r["check1"]["degenerate_full_entropy"] is False

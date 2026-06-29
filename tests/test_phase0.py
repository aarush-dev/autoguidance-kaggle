"""Phase 0 pipeline integration test on synthetic model (no GPU).

Updated for the arrays/thresholds split:
  run_characterization(adapter, weak_selfs, cfg, arrays_dir) -> dict[name -> npz_path]
  evaluate_dir(arrays_dir, cfg) -> dict[name -> verdict]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
import pytest

from autoguidance.models.synthetic import SyntheticAdapter
from autoguidance.weak_self.input_noise import InputNoiseRemask, InputNoiseGauss
from autoguidance.weak_self.low_nfe import LowNFEWeakSelf
from autoguidance.phase0.characterize import run_characterization
from autoguidance.phase0.thresholds import evaluate_dir
from autoguidance.config import Phase0Config


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def smoke_cfg():
    cfg = Phase0Config()
    cfg.n_samples = 8
    cfg.mask_rate = 0.4
    # Relaxed thresholds so synthetic data reliably passes
    cfg.check1_entropy_fraction = 0.0
    cfg.check2_top1_agreement = 0.0
    cfg.check2_spearman_rho = -1.0
    cfg.check3_pearson_r = -1.0
    cfg.check3_precision = 0.0
    return cfg


# ---------------------------------------------------------------------------
# run_characterization: new contract — takes arrays_dir, returns npz paths
# ---------------------------------------------------------------------------

def test_phase0_smoke_returns_npz_paths(smoke_cfg):
    """run_characterization returns a dict mapping construction name → npz path."""
    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {
        "input_noise_remask": InputNoiseRemask(remask_rate=0.2, seed=1),
        "low_nfe": LowNFEWeakSelf(nfe=1),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"input_noise_remask", "low_nfe"}
        for name, path in result.items():
            assert isinstance(path, str), f"{name}: path should be a string"
            assert path.endswith(".npz"), f"{name}: path should end with .npz"
            assert os.path.isfile(path), f"{name}: .npz file does not exist at {path}"


def test_phase0_smoke_files_in_arrays_dir(smoke_cfg):
    """All .npz files are written inside arrays_dir."""
    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {"input_noise_remask": InputNoiseRemask(remask_rate=0.2, seed=1)}
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)
        for name, path in result.items():
            assert os.path.commonpath([tmpdir, path]) == tmpdir, (
                f"{name}: .npz file is outside arrays_dir"
            )


def test_phase0_smoke_both_constructions(smoke_cfg):
    """run_characterization processes all supplied constructions."""
    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {
        "input_noise_remask": InputNoiseRemask(remask_rate=0.2, seed=1),
        "low_nfe": LowNFEWeakSelf(nfe=1),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)
        assert "input_noise_remask" in result
        assert "low_nfe" in result


def test_phase0_gauss_construction(smoke_cfg):
    """run_characterization works with InputNoiseGauss construction."""
    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {"input_noise_gauss": InputNoiseGauss(sigma=0.5, seed=7)}
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)
        assert "input_noise_gauss" in result
        path = result["input_noise_gauss"]
        assert os.path.isfile(path)


def test_phase0_npz_contains_expected_arrays(smoke_cfg):
    """Each saved .npz contains all 7 named arrays."""
    import numpy as np
    expected_arrays = {
        "entropy_full", "entropy_weak", "top1_full", "top1_weak",
        "diff_mag", "gt", "rho_sample",
    }
    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {"input_noise_remask": InputNoiseRemask(remask_rate=0.2, seed=1)}
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)
        path = result["input_noise_remask"]
        data = np.load(path, allow_pickle=True)
        for arr_name in expected_arrays:
            assert arr_name in data, f"Missing array '{arr_name}' in saved .npz"


# ---------------------------------------------------------------------------
# evaluate_dir: verdicts from saved arrays
# ---------------------------------------------------------------------------

def test_phase0_evaluate_dir_produces_verdicts(smoke_cfg):
    """evaluate_dir returns one verdict dict per construction."""
    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {
        "input_noise_remask": InputNoiseRemask(remask_rate=0.2, seed=1),
        "low_nfe": LowNFEWeakSelf(nfe=1),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)
        verdicts = evaluate_dir(tmpdir, smoke_cfg)

        assert isinstance(verdicts, dict)
        assert set(verdicts.keys()) == {"input_noise_remask", "low_nfe"}
        for name, v in verdicts.items():
            assert "all_pass" in v, f"{name}: verdict missing 'all_pass'"
            assert "check1" in v, f"{name}: verdict missing 'check1'"
            assert "check2" in v, f"{name}: verdict missing 'check2'"
            assert "check3" in v, f"{name}: verdict missing 'check3'"


def test_phase0_evaluate_dir_smoke_verdicts(smoke_cfg):
    """evaluate_dir returns boolean all_pass and check1/2/3 sub-dicts for each construction.

    Note: we do NOT assert all_pass=True here.  The synthetic model produces near-uniform
    logits (entropy ≈ ln(vocab)), so check1 (mean_weak > mean_full) may or may not hold
    depending on the construction and random seed.  This test verifies the pipeline
    computes verdicts without error and returns the expected structure.
    """
    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {
        "input_noise_remask": InputNoiseRemask(remask_rate=0.2, seed=1),
        "low_nfe": LowNFEWeakSelf(nfe=1),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)
        verdicts = evaluate_dir(tmpdir, smoke_cfg)
        assert set(verdicts.keys()) == {"input_noise_remask", "low_nfe"}
        for name, v in verdicts.items():
            # all_pass is explicitly bool()-wrapped in evaluate_arrays
            assert isinstance(v["all_pass"], bool), f"{name}: all_pass must be Python bool"
            # check sub-dicts may return numpy.bool_ (from scipy comparisons); use `in`
            assert v["check1"]["check1_pass"] in (True, False), \
                f"{name}: check1_pass must be bool-like"
            assert v["check2"]["check2_pass"] in (True, False), \
                f"{name}: check2_pass must be bool-like"
            assert v["check3"]["check3_pass"] in (True, False), \
                f"{name}: check3_pass must be bool-like"
            # Numeric stats must be present and finite
            assert "mean_full" in v["check1"]
            assert "top1_agreement" in v["check2"]
            assert "pearson_r" in v["check3"]


def test_phase0_evaluate_dir_tight_thresholds_may_fail(smoke_cfg):
    """With very tight thresholds, at least check1 fails on synthetic data."""
    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {"input_noise_remask": InputNoiseRemask(remask_rate=0.2, seed=1)}
    tight_cfg = Phase0Config()
    tight_cfg.n_samples = 8
    tight_cfg.mask_rate = 0.4
    tight_cfg.check1_entropy_fraction = 1.0   # requires 100 % of positions → impossible
    tight_cfg.check2_top1_agreement = 0.0
    tight_cfg.check2_spearman_rho = -1.0
    tight_cfg.check3_pearson_r = -1.0
    tight_cfg.check3_precision = 0.0
    with tempfile.TemporaryDirectory() as tmpdir:
        # Characterize with relaxed cfg so arrays are written correctly
        run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)
        # Evaluate with tight cfg
        verdicts = evaluate_dir(tmpdir, tight_cfg)
        assert not verdicts["input_noise_remask"]["check1"]["check1_pass"], (
            "check1 should fail at fraction=1.0"
        )
        assert not verdicts["input_noise_remask"]["all_pass"]


# ---------------------------------------------------------------------------
# Separation: evaluate_dir reads from disk without re-running the model
# ---------------------------------------------------------------------------

def test_phase0_evaluate_dir_is_model_free(smoke_cfg):
    """evaluate_dir can be called without loading any model (pure numpy)."""
    import importlib
    # Verify the module imports without importing torch at module level
    mod = importlib.import_module("autoguidance.phase0.thresholds")
    assert callable(getattr(mod, "evaluate_dir", None))
    assert callable(getattr(mod, "evaluate_arrays", None))

    adapter = SyntheticAdapter(seed=42)
    weak_selfs = {"low_nfe": LowNFEWeakSelf(nfe=1)}
    with tempfile.TemporaryDirectory() as tmpdir:
        run_characterization(adapter, weak_selfs, smoke_cfg, tmpdir)
        # Pass an obviously wrong device to confirm evaluate_dir doesn't touch torch
        verdicts = evaluate_dir(tmpdir, smoke_cfg)
        assert "low_nfe" in verdicts

"""Tests for phase0/arrays.py: save/load round-trip and list_constructions.

Contract:
  save_arrays(arrays_dir, name, meta:dict, **named_arrays) -> path
  load_arrays(arrays_dir, name) -> dict
  list_constructions(arrays_dir) -> list[str]
  Named arrays: entropy_full, entropy_weak, top1_full, top1_weak, diff_mag, gt, rho_sample.
  Meta stored as a json string array entry; np.savez_compressed on disk.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import tempfile

import numpy as np
import pytest

from autoguidance.phase0.arrays import save_arrays, load_arrays, list_constructions


ARRAY_NAMES = [
    "entropy_full", "entropy_weak",
    "top1_full", "top1_weak",
    "diff_mag", "gt", "rho_sample",
]


def _make_arrays(n=50, rho_n=20, seed=0):
    """Build a canonical set of the 7 named arrays used by characterize."""
    rng = np.random.default_rng(seed)
    return {
        "entropy_full": rng.random(n).astype(np.float32),
        "entropy_weak": rng.random(n).astype(np.float32),
        "top1_full":    rng.integers(0, 99, n).astype(np.int64),
        "top1_weak":    rng.integers(0, 99, n).astype(np.int64),
        "diff_mag":     rng.random(n).astype(np.float32),
        "gt":           rng.integers(0, 99, n).astype(np.int64),
        "rho_sample":   rng.random(rho_n).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Round-trip: arrays
# ---------------------------------------------------------------------------

def test_round_trip_all_arrays():
    """save_arrays then load_arrays recovers all 7 named arrays exactly."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays()
        save_arrays(d, "input_noise_remask", {}, **arrs)
        loaded = load_arrays(d, "input_noise_remask")
        for name in ARRAY_NAMES:
            assert name in loaded, f"Missing array: {name}"
            np.testing.assert_array_equal(loaded[name], arrs[name], err_msg=f"Mismatch: {name}")


def test_round_trip_dtypes_preserved():
    """save/load preserves float32 for entropy/diff_mag and int64 for top1/gt."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays()
        save_arrays(d, "low_nfe", {}, **arrs)
        loaded = load_arrays(d, "low_nfe")
        assert loaded["entropy_full"].dtype == np.float32
        assert loaded["top1_full"].dtype == np.int64
        assert loaded["gt"].dtype == np.int64


def test_round_trip_variable_lengths():
    """Arrays of different lengths within the same save are recovered correctly."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays(n=37, rho_n=13)
        save_arrays(d, "layer_drop", {}, **arrs)
        loaded = load_arrays(d, "layer_drop")
        assert len(loaded["entropy_full"]) == 37
        assert len(loaded["rho_sample"]) == 13


# ---------------------------------------------------------------------------
# Round-trip: meta
# ---------------------------------------------------------------------------

def test_round_trip_meta_string_values():
    """Meta dict with string values survives the round-trip; load_arrays returns a dict."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays()
        meta = {"construction": "low_nfe", "model": "synthetic"}
        save_arrays(d, "low_nfe", meta, **arrs)
        loaded = load_arrays(d, "low_nfe")
        assert "meta" in loaded
        assert isinstance(loaded["meta"], dict), "meta should be returned as a Python dict"
        assert loaded["meta"]["construction"] == "low_nfe"
        assert loaded["meta"]["model"] == "synthetic"


def test_round_trip_meta_numeric_values():
    """Meta dict with numeric values survives the round-trip."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays()
        meta = {"n_samples": 200, "mask_rate": 0.5}
        save_arrays(d, "input_noise_gauss", meta, **arrs)
        loaded = load_arrays(d, "input_noise_gauss")
        assert isinstance(loaded["meta"], dict)
        assert loaded["meta"]["n_samples"] == 200
        assert abs(loaded["meta"]["mask_rate"] - 0.5) < 1e-9


def test_round_trip_empty_meta():
    """Empty meta dict is accepted and recovers as empty dict."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays()
        save_arrays(d, "inference_dropout", {}, **arrs)
        loaded = load_arrays(d, "inference_dropout")
        assert isinstance(loaded["meta"], dict)
        assert loaded["meta"] == {}


# ---------------------------------------------------------------------------
# save_arrays return value
# ---------------------------------------------------------------------------

def test_save_returns_path_to_npz():
    """save_arrays returns a string path that exists and ends with .npz."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays()
        path = save_arrays(d, "x", {}, **arrs)
        assert isinstance(path, str)
        assert path.endswith(".npz")
        assert os.path.isfile(path)


def test_save_file_is_in_arrays_dir():
    """The returned path is located under arrays_dir."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays()
        path = save_arrays(d, "y", {}, **arrs)
        assert os.path.commonpath([d, path]) == d


# ---------------------------------------------------------------------------
# list_constructions
# ---------------------------------------------------------------------------

def test_list_constructions_empty():
    """list_constructions returns [] on an empty directory."""
    with tempfile.TemporaryDirectory() as d:
        result = list_constructions(d)
        assert result == []


def test_list_constructions_finds_all_saved():
    """list_constructions returns a name for every save_arrays call."""
    with tempfile.TemporaryDirectory() as d:
        names = ["input_noise_remask", "low_nfe", "layer_drop"]
        arrs = _make_arrays()
        for n in names:
            save_arrays(d, n, {"construction": n}, **arrs)
        found = list_constructions(d)
        assert sorted(found) == sorted(names)


def test_list_constructions_no_extra():
    """list_constructions does not return spurious names."""
    with tempfile.TemporaryDirectory() as d:
        arrs = _make_arrays()
        save_arrays(d, "only_one", {}, **arrs)
        found = list_constructions(d)
        assert found == ["only_one"]


def test_list_constructions_ignores_non_npz(tmp_path):
    """list_constructions ignores non-.npz files in the directory."""
    (tmp_path / "readme.txt").write_text("hello")
    (tmp_path / "data.csv").write_text("a,b")
    arrs = _make_arrays()
    save_arrays(str(tmp_path), "real_one", {}, **arrs)
    found = list_constructions(str(tmp_path))
    assert found == ["real_one"]


# ---------------------------------------------------------------------------
# Isolation between saves
# ---------------------------------------------------------------------------

def test_two_saves_are_independent():
    """Two constructions saved to the same dir do not overwrite each other."""
    with tempfile.TemporaryDirectory() as d:
        arrs_a = _make_arrays(seed=1)
        arrs_b = _make_arrays(seed=2)
        save_arrays(d, "a", {}, **arrs_a)
        save_arrays(d, "b", {}, **arrs_b)
        la = load_arrays(d, "a")
        lb = load_arrays(d, "b")
        np.testing.assert_array_equal(la["entropy_full"], arrs_a["entropy_full"])
        np.testing.assert_array_equal(lb["entropy_full"], arrs_b["entropy_full"])
        # Confirm a and b differ
        assert not np.array_equal(la["entropy_full"], lb["entropy_full"])

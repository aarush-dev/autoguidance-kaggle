"""Tests for eval modules (no GPU needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import pytest

from autoguidance.eval.entropy import compute_entropy_stats
from autoguidance.eval.agreement import compute_agreement_stats
from autoguidance.eval.error_position import compute_error_position_stats
from autoguidance.eval.distinct_n import distinct_n
from autoguidance.eval.self_bleu import self_bleu
from autoguidance.eval.task_accuracy import make_sudoku_batch, sudoku_accuracy, _is_valid_sudoku


def make_logits(B=2, L=10, V=50, seed=0):
    rng = torch.manual_seed(seed)
    return torch.randn(B, L, V)


def make_mask(B=2, L=10, rate=0.5, seed=0):
    np.random.seed(seed)
    mask = np.random.rand(B, L) < rate
    return mask.astype(bool)


# --- entropy ---

def test_entropy_shape():
    full = make_logits()
    weak = make_logits(seed=1)
    mask = make_mask()
    r = compute_entropy_stats(full, weak, mask)
    assert "full_entropy" in r
    assert "check1_pass" in r
    assert len(r["full_entropy"]) == mask.sum()


def test_entropy_higher_entropy_noise():
    """Adding large noise to logits should increase entropy."""
    B, L, V = 2, 10, 50
    full = torch.zeros(B, L, V)  # uniform → max entropy already
    # For this test, use peaked logits as full and noisy as weak
    peaked = torch.zeros(B, L, V)
    peaked[:, :, 0] = 10.0     # very peaked → low entropy
    noisy = torch.randn(B, L, V)  # noisy → higher entropy
    mask = np.ones((B, L), dtype=bool)
    r = compute_entropy_stats(peaked, noisy, mask, check1_entropy_fraction=0.0)
    # noisy logits should have higher entropy than peaked
    assert r["mean_weak"] > r["mean_full"]


def test_entropy_check1_pass_criterion():
    B, L, V = 1, 20, 50
    peaked = torch.zeros(B, L, V)
    peaked[:, :, 0] = 20.0    # very peaked
    noisy = torch.randn(B, L, V)
    mask = np.ones((B, L), dtype=bool)
    r = compute_entropy_stats(peaked, noisy, mask, check1_entropy_fraction=0.0)
    assert r["check1_pass"] is True


# --- agreement ---

def test_agreement_perfect():
    """Identical logits → top1 agreement = 1.0."""
    logits = make_logits(B=2, L=10, V=50, seed=0)
    mask = make_mask()
    r = compute_agreement_stats(logits, logits, mask,
                                 check2_top1_agreement=1.0,
                                 check2_spearman_rho=1.0)
    assert abs(r["top1_agreement"] - 1.0) < 1e-6
    assert r["spearman_rho_mean"] > 0.99


def test_agreement_random():
    full = make_logits(B=2, L=10, V=50, seed=0)
    weak = make_logits(B=2, L=10, V=50, seed=99)
    mask = make_mask()
    r = compute_agreement_stats(full, weak, mask,
                                 check2_top1_agreement=0.0,
                                 check2_spearman_rho=-1.0)
    assert 0.0 <= r["top1_agreement"] <= 1.0
    assert -1.0 <= r["spearman_rho_mean"] <= 1.0


# --- error_position ---

def test_error_position_shape():
    full = make_logits()
    weak = make_logits(seed=1)
    gt = torch.randint(0, 50, (2, 10))
    mask = make_mask()
    r = compute_error_position_stats(full, weak, gt, mask)
    assert "pearson_r" in r
    assert "check3_pass" in r


def test_error_position_perfect_agreement():
    """If weak=full, disagreement=0, error_position correlation = 0."""
    logits = make_logits()
    gt = torch.randint(0, 50, (2, 10))
    mask = make_mask()
    r = compute_error_position_stats(logits, logits, gt, mask)
    assert r["disagreement_rate"] < 1e-6


# --- distinct_n ---

def test_distinct_1_diverse():
    texts = ["the cat sat", "a dog ran", "some bird flew", "fish swim fast"]
    assert distinct_n(texts, 1) > 0.5


def test_distinct_1_collapsed():
    texts = ["cat cat cat", "cat cat cat", "cat cat cat"]
    assert distinct_n(texts, 1) < 0.2   # 3 texts × "cat cat cat" → 1 unique / 9 total = 0.111


def test_distinct_2_empty():
    assert distinct_n([], 2) == 0.0


# --- self_bleu ---

def test_self_bleu_diverse():
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "python is a popular programming language for data science",
        "the universe is approximately 13.8 billion years old",
    ]
    sb = self_bleu(texts, n_pairs=4, seed=42)
    assert 0.0 <= sb <= 1.0


def test_self_bleu_identical():
    texts = ["hello world how are you"] * 5
    sb = self_bleu(texts, n_pairs=5, seed=0)
    # Should be close to 1 for near-identical texts
    assert sb > 0.5


# --- sudoku ---

def test_sudoku_generation():
    prompts, solutions = make_sudoku_batch(3, seed=0)
    assert len(prompts) == 3
    assert len(solutions) == 3
    for sol in solutions:
        assert _is_valid_sudoku(sol), "Generated solution should be valid"


def test_sudoku_accuracy_perfect():
    prompts, solutions = make_sudoku_batch(2, seed=0)
    # Perfect outputs: provide the solution as text
    perfect_outputs = []
    for sol in solutions:
        flat = [str(v) for row in sol for v in row]
        perfect_outputs.append(" ".join(flat))
    acc = sudoku_accuracy(perfect_outputs, solutions)
    assert acc == 1.0


def test_sudoku_accuracy_wrong():
    _, solutions = make_sudoku_batch(2, seed=0)
    wrong = ["no digits here", "also no digits"]
    acc = sudoku_accuracy(wrong, solutions)
    assert acc == 0.0

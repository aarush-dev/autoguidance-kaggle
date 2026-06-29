"""Distinct-n diversity metrics for Phase 1."""
from __future__ import annotations
from typing import List


def distinct_n(texts: List[str], n: int) -> float:
    """Fraction of unique n-grams across all generated texts.

    Higher = more diverse. Range [0, 1].
    """
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

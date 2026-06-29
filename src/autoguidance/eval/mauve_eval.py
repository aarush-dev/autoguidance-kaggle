"""MAUVE diversity metric for Phase 1.

MAUVE measures how close generated text is to human-text references.
High MAUVE = distribution close to human text = diverse, natural outputs.
Low MAUVE = mode collapse / degenerate text.

Runs on CUDA:1 (Quadro P4000, fp16). No bitsandbytes on Pascal.
"""
from __future__ import annotations
from typing import List
import numpy as np


def compute_mauve(
    generated_texts: List[str],
    reference_texts: List[str],
    device_id: int = 1,
    max_text: int = 1000,
    featurize_model_name: str = "gpt2-large",
) -> float:
    """Compute MAUVE score.

    Args:
        generated_texts: model outputs.
        reference_texts: human reference texts (e.g., WikiText-103 validation).
        device_id: CUDA device index for feature extraction (use 1 for P4000).
        max_text: cap number of texts for speed.
        featurize_model_name: model for embedding features.

    Returns:
        MAUVE score in [0, 1] (higher = more human-like = more diverse).
    """
    import evaluate

    gen = generated_texts[:max_text]
    ref = reference_texts[:max_text]

    # Pad to same length if needed
    n = min(len(gen), len(ref))
    if n < 2:
        return 0.0
    gen, ref = gen[:n], ref[:n]

    mauve_metric = evaluate.load("mauve")
    result = mauve_metric.compute(
        predictions=gen,
        references=ref,
        device_id=device_id,
        featurize_model_name=featurize_model_name,
    )
    return float(result.mauve)

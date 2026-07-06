"""Self-BLEU diversity metric for Phase 1.

Lower self-BLEU = more diverse outputs. Measures average BLEU of each generated
text against the rest of the generation set.
"""
from __future__ import annotations
from typing import List
import random

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# punkt is expected under NLTK_DATA (bundled offline in the hfdata mount). Only
# attempt a download if it is genuinely missing, and never let that break import
# in an offline environment.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


def self_bleu(texts: List[str], n_pairs: int = 100, seed: int = 42) -> float:
    """Compute self-BLEU over a sample of pairs.

    Args:
        texts: generated text strings.
        n_pairs: number of (hypothesis, references) pairs to sample.
        seed: random seed for sampling.

    Returns:
        mean BLEU score (lower = more diverse).
    """
    if len(texts) < 2:
        return 0.0

    rng = random.Random(seed)
    smoother = SmoothingFunction().method1
    scores = []

    for _ in range(min(n_pairs, len(texts))):
        hyp_idx = rng.randrange(len(texts))
        hyp = texts[hyp_idx].split()
        refs = [texts[i].split() for i in range(len(texts)) if i != hyp_idx]
        if not refs:
            continue
        score = sentence_bleu(refs, hyp, smoothing_function=smoother)
        scores.append(score)

    return float(sum(scores) / len(scores)) if scores else 0.0

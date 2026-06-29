"""Generative perplexity scorer for Phase 1.

Scores generated text with GPT-2-large on the Quadro P4000 (CUDA:1).
No bitsandbytes kernels — Pascal arch does not support them.
Uses fp16 to fit in 8 GB VRAM.
"""
from __future__ import annotations
from typing import List

import torch
import numpy as np


_scorer_cache: dict = {}


def _get_scorer(model_name: str, device: str):
    key = (model_name, device)
    if key not in _scorer_cache:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device).eval()
        _scorer_cache[key] = (tok, model)
    return _scorer_cache[key]


@torch.no_grad()
def generative_perplexity(
    texts: List[str],
    scorer_model: str = "gpt2-large",
    device: str = "cuda:1",
    max_length: int = 256,
) -> float:
    """Mean generative perplexity of texts under the scorer model.

    Lower perplexity = higher quality. Runs on CUDA:1 (P4000, fp16).

    Args:
        texts: generated text strings.
        scorer_model: HuggingFace model ID for scoring.
        device: device for scorer (must NOT be the device running LLaDA).
        max_length: truncate texts to this many tokens.

    Returns:
        mean perplexity (float).
    """
    if not texts:
        return float("inf")

    tok, model = _get_scorer(scorer_model, device)
    ppls = []

    for text in texts:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        # out.loss is the mean NLL over tokens
        ppls.append(torch.exp(out.loss).item())

    return float(np.mean(ppls)) if ppls else float("inf")

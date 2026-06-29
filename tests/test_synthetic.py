"""Smoke tests for the ModelAdapter interface on the synthetic model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from autoguidance.models.synthetic import SyntheticAdapter, VOCAB_SIZE, SEQ_LEN, MASK_ID
from autoguidance.config import Phase0Config


@pytest.fixture
def adapter():
    return SyntheticAdapter(seed=42)


def test_mask_token_id(adapter):
    assert adapter.mask_token_id == MASK_ID


def test_vocab_size(adapter):
    assert adapter.vocab_size == VOCAB_SIZE


def test_logits_shape(adapter):
    B, L = 2, SEQ_LEN
    x = torch.randint(0, VOCAB_SIZE - 1, (B, L))  # no MASK tokens
    logits = adapter.logits(x)
    assert logits.shape == (B, L, VOCAB_SIZE)


def test_logits_with_mask(adapter):
    B, L = 1, SEQ_LEN
    x = torch.full((B, L), MASK_ID, dtype=torch.long)
    logits = adapter.logits(x)
    assert logits.shape == (B, L, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_embed_shape(adapter):
    B, L = 2, SEQ_LEN
    x = torch.randint(0, VOCAB_SIZE, (B, L))
    embeds = adapter.embed(x)
    assert embeds.shape[0] == B
    assert embeds.shape[1] == L
    assert embeds.dim() == 3


def test_logits_from_embed(adapter):
    B, L = 2, SEQ_LEN
    x = torch.randint(0, VOCAB_SIZE, (B, L))
    embeds = adapter.embed(x)
    logits = adapter.logits_from_embed(embeds)
    assert logits.shape == (B, L, VOCAB_SIZE)


def test_logits_from_embed_matches_logits_approximately(adapter):
    """logits(x) and logits_from_embed(embed(x)) should agree exactly (same path)."""
    B, L = 1, SEQ_LEN
    x = torch.randint(0, VOCAB_SIZE, (B, L))
    logits_a = adapter.logits(x)
    embeds = adapter.embed(x)
    logits_b = adapter.logits_from_embed(embeds)
    assert torch.allclose(logits_a, logits_b, atol=1e-5), \
        "logits and logits_from_embed should agree on the same input"


def test_encode_decode(adapter):
    text = "hello world foo bar"
    ids = adapter.encode(text)
    assert ids.dim() == 2
    assert ids.shape[0] == 1
    decoded = adapter.decode(ids)
    assert isinstance(decoded, str)


def test_deterministic(adapter):
    x = torch.tensor([[1, 2, 3, MASK_ID, 5]], dtype=torch.long)
    out1 = adapter.logits(x)
    out2 = adapter.logits(x)
    assert torch.allclose(out1, out2)

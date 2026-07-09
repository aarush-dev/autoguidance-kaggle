"""Tests for MoE patching utilities and ReducedExpertWeakSelf.

Contract (autoguidance.weak_self.moe_patch + autoguidance.weak_self.reduced_expert):
  list_moe_modules(model: nn.Module) -> None   — prints found MoE submodules (debug)
  reduce_moe_topk(model, new_top_k) -> context manager — set top_k inside, restore after
  ReducedExpertWeakSelf(top_k=4):
    - raises NotImplementedError on adapter.is_moe == False
    - works (returns [B,L,V] logits, restores top_k) on adapter.is_moe == True
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from autoguidance.weak_self.moe_patch import reduce_moe_topk, list_moe_modules
from autoguidance.weak_self.reduced_expert import ReducedExpertWeakSelf
from autoguidance.models.synthetic import SyntheticAdapter, VOCAB_SIZE, SEQ_LEN, MASK_ID


# ---------------------------------------------------------------------------
# Minimal fixture: a module with and without top_k
# ---------------------------------------------------------------------------

class _FakeRouter(nn.Module):
    """Simulates a MoE routing block: has top_k int attribute."""
    def __init__(self, top_k: int = 8):
        super().__init__()
        self.top_k = top_k
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


class _FakeDenseBlock(nn.Module):
    """Dense block — no top_k attribute."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


class _FakeMoEModel(nn.Module):
    """Tiny 'MoE' model: one routing layer + one dense layer."""
    def __init__(self):
        super().__init__()
        self.router = _FakeRouter(top_k=8)
        self.dense = _FakeDenseBlock()

    def forward(self, x):
        return self.dense(self.router(x))


# ---------------------------------------------------------------------------
# list_moe_modules: debug printer (returns None, no return value to assert on)
# ---------------------------------------------------------------------------

def test_list_moe_modules_does_not_raise():
    """list_moe_modules completes without error on a model with top_k attributes."""
    model = _FakeMoEModel()
    list_moe_modules(model)   # should not raise


def test_list_moe_modules_no_crash_on_plain_model():
    """list_moe_modules handles models with no top_k attributes (prints 'none found')."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    list_moe_modules(model)   # should not raise


def test_list_moe_modules_prints_router(capsys):
    """list_moe_modules prints the routing layer's top_k value."""
    model = _FakeMoEModel()
    list_moe_modules(model)
    out = capsys.readouterr().out
    assert "top_k" in out, f"Expected 'top_k' in output:\n{out}"
    assert "8" in out, f"Expected value '8' in output:\n{out}"


def test_list_moe_modules_plain_model_reports_none_found(capsys):
    """list_moe_modules reports that no attributes were found for a plain model."""
    model = nn.Sequential(nn.Linear(4, 4))
    list_moe_modules(model)
    out = capsys.readouterr().out
    # Should mention that nothing was found
    assert "no" in out.lower() or "0" in out, f"Expected 'no/0 found' message:\n{out}"


# ---------------------------------------------------------------------------
# reduce_moe_topk: context manager
# ---------------------------------------------------------------------------

def test_reduce_moe_topk_sets_inside_context():
    """Inside the context, top_k is set to new_top_k (4 < 8 triggers the patch)."""
    model = _FakeMoEModel()
    assert model.router.top_k == 8
    with reduce_moe_topk(model, new_top_k=4):
        assert model.router.top_k == 4, f"Expected 4, got {model.router.top_k}"


def test_reduce_moe_topk_restores_after_context():
    """After the context exits normally, top_k is restored to original value."""
    model = _FakeMoEModel()
    with reduce_moe_topk(model, new_top_k=4):
        pass
    assert model.router.top_k == 8, f"Expected 8, got {model.router.top_k}"


def test_reduce_moe_topk_restores_on_exception():
    """top_k is restored even if an exception is raised inside the context."""
    model = _FakeMoEModel()
    try:
        with reduce_moe_topk(model, new_top_k=2):
            assert model.router.top_k == 2
            raise ValueError("simulated failure")
    except ValueError:
        pass
    assert model.router.top_k == 8, "top_k not restored after exception"


def test_reduce_moe_topk_skips_if_new_not_smaller():
    """reduce_moe_topk is a no-op when new_top_k >= existing value (safety guard)."""
    model = _FakeMoEModel()
    # new_top_k=8 equals the existing value (8 >= 8) → no patch per contract
    with reduce_moe_topk(model, new_top_k=8):
        # top_k should remain 8 (either unchanged or skipped)
        assert model.router.top_k == 8
    assert model.router.top_k == 8


def test_reduce_moe_topk_handles_multiple_routers():
    """reduce_moe_topk patches every matching router in the model."""
    class _MultiRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.r1 = _FakeRouter(top_k=8)
            self.r2 = _FakeRouter(top_k=8)
    model = _MultiRouter()
    with reduce_moe_topk(model, new_top_k=3):
        assert model.r1.top_k == 3
        assert model.r2.top_k == 3
    assert model.r1.top_k == 8
    assert model.r2.top_k == 8


class _FakeConfig:
    """Simulates DiffusionGemmaTextConfig: top_k_experts lives on config, not
    on any router nn.Module."""
    def __init__(self, top_k_experts: int = 8):
        self.top_k_experts = top_k_experts


class _ConfigOnlyMoEModel(nn.Module):
    """Model whose MoE top-k is config-level only (no cached module attr) —
    mirrors DiffusionGemma's real layout."""
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig(top_k_experts=8)
        self.dense = _FakeDenseBlock()


def test_reduce_moe_topk_patches_config_level_top_k():
    """DiffusionGemma-style models keep top_k_experts on model.config, not on
    a router module. reduce_moe_topk must still find and patch it."""
    model = _ConfigOnlyMoEModel()
    assert model.config.top_k_experts == 8
    with reduce_moe_topk(model, new_top_k=4):
        assert model.config.top_k_experts == 4, (
            f"Expected config.top_k_experts patched to 4, got {model.config.top_k_experts}"
        )
    assert model.config.top_k_experts == 8, "config.top_k_experts not restored"


def test_list_moe_modules_finds_config_level_top_k(capsys):
    """list_moe_modules must report top_k_experts even when it only exists on
    model.config (not any nn.Module)."""
    model = _ConfigOnlyMoEModel()
    list_moe_modules(model)
    out = capsys.readouterr().out
    assert "top_k_experts" in out, f"Expected 'top_k_experts' in output:\n{out}"
    assert "8" in out, f"Expected value '8' in output:\n{out}"


def test_reduce_moe_topk_restores_heterogeneous_values():
    """Each router is restored to its own original top_k, not a shared value."""
    class _HeteroRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.r1 = _FakeRouter(top_k=8)
            self.r2 = _FakeRouter(top_k=6)
    model = _HeteroRouter()
    with reduce_moe_topk(model, new_top_k=2):
        assert model.r1.top_k == 2
        assert model.r2.top_k == 2
    assert model.r1.top_k == 8, f"r1: expected 8, got {model.r1.top_k}"
    assert model.r2.top_k == 6, f"r2: expected 6, got {model.r2.top_k}"


# ---------------------------------------------------------------------------
# ReducedExpertWeakSelf on non-MoE adapter
# ---------------------------------------------------------------------------

def test_reduced_expert_raises_on_non_moe():
    """ReducedExpertWeakSelf raises NotImplementedError when adapter.is_moe is False."""
    adapter = SyntheticAdapter(seed=42)
    assert not adapter.is_moe, "Precondition: SyntheticAdapter.is_moe should be False"
    x = torch.randint(0, VOCAB_SIZE - 1, (1, SEQ_LEN))
    ws = ReducedExpertWeakSelf(top_k=4)
    with pytest.raises(NotImplementedError):
        ws(x, None, adapter)


# ---------------------------------------------------------------------------
# ReducedExpertWeakSelf on MoE-sim adapter
# ---------------------------------------------------------------------------

class _MoESyntheticAdapter(SyntheticAdapter):
    """SyntheticAdapter variant with is_moe=True and MoE-like layers.

    The underlying _TinyMaskedLM encoder layers get a fake top_k=8 attribute so
    reduce_moe_topk / list_moe_modules can find and patch them.  The forward pass
    is unchanged (the synthetic model doesn't gate on top_k), which is fine —
    we just test that ReducedExpertWeakSelf does NOT raise and returns the correct shape.
    """
    def __init__(self):
        super().__init__(seed=99)
        # Inject top_k onto each transformer encoder layer.
        for layer in self._model.encoder.layers:
            layer.top_k = 8

    @property
    def is_moe(self) -> bool:
        return True


def _make_x(B=2):
    x = torch.randint(0, VOCAB_SIZE - 1, (B, SEQ_LEN))
    x[:, 5:10] = MASK_ID
    return x


def test_reduced_expert_works_on_moe_adapter():
    """ReducedExpertWeakSelf returns [B,L,V] logits on an is_moe=True adapter."""
    adapter = _MoESyntheticAdapter()
    x = _make_x()
    ws = ReducedExpertWeakSelf(top_k=4)
    logits = ws(x, None, adapter)
    assert logits.shape == (x.shape[0], SEQ_LEN, VOCAB_SIZE), (
        f"Expected ({x.shape[0]}, {SEQ_LEN}, {VOCAB_SIZE}), got {logits.shape}"
    )


def test_reduced_expert_restores_top_k_after_call():
    """After the weak pass, all MoE layers have their original top_k restored."""
    adapter = _MoESyntheticAdapter()
    x = _make_x(B=1)
    ws = ReducedExpertWeakSelf(top_k=4)
    ws(x, None, adapter)
    for layer in adapter._model.encoder.layers:
        assert layer.top_k == 8, f"top_k not restored; got {layer.top_k}"


def test_reduced_expert_restores_top_k_on_forward_error():
    """If the forward pass raises, top_k is still restored via context manager."""
    class _BrokenMoEAdapter(_MoESyntheticAdapter):
        def logits(self, x_t, cond=None):
            raise RuntimeError("simulated forward error")

    adapter = _BrokenMoEAdapter()
    x = _make_x(B=1)
    ws = ReducedExpertWeakSelf(top_k=4)
    with pytest.raises(RuntimeError):
        ws(x, None, adapter)
    for layer in adapter._model.encoder.layers:
        assert layer.top_k == 8, f"top_k not restored after error; got {layer.top_k}"

"""Tests for weak-self constructions on the synthetic model."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional
import pytest

from autoguidance.models.synthetic import SyntheticAdapter, VOCAB_SIZE, SEQ_LEN, MASK_ID
from autoguidance.weak_self.input_noise import InputNoiseRemask, InputNoiseGauss
from autoguidance.weak_self.low_nfe import LowNFEWeakSelf
from autoguidance.weak_self.reduced_expert import ReducedExpertWeakSelf
from autoguidance.weak_self.layer_drop import LayerDropWeakSelf
from autoguidance.weak_self.inference_dropout import InferenceDropoutWeakSelf
from autoguidance.weak_self import load_weak_self
from autoguidance.config import Phase0Config


@pytest.fixture
def adapter():
    return SyntheticAdapter(seed=42)


def make_x(B=2):
    x = torch.randint(0, VOCAB_SIZE - 1, (B, SEQ_LEN))
    # mask some positions
    x[:, 5:10] = MASK_ID
    return x


# ---------------------------------------------------------------------------
# Existing tests — InputNoiseRemask
# ---------------------------------------------------------------------------

def test_remask_output_shape(adapter):
    x = make_x()
    ws = InputNoiseRemask(remask_rate=0.2, seed=1)
    logits = ws(x, None, adapter)
    assert logits.shape == (x.shape[0], SEQ_LEN, VOCAB_SIZE)


def test_remask_changes_some_tokens():
    """Re-masking should produce different logits from the full model pass."""
    adapter = SyntheticAdapter(seed=42)
    x = torch.randint(0, VOCAB_SIZE - 1, (1, SEQ_LEN))  # no masked positions
    full_logits = adapter.logits(x)
    ws = InputNoiseRemask(remask_rate=0.5, seed=99)
    weak_logits = ws(x, None, adapter)
    # They should differ (re-masking changes input → different logits)
    assert not torch.allclose(full_logits, weak_logits)


# ---------------------------------------------------------------------------
# Existing tests — InputNoiseGauss
# ---------------------------------------------------------------------------

def test_gauss_output_shape(adapter):
    x = make_x()
    ws = InputNoiseGauss(sigma=0.5, seed=1)
    logits = ws(x, None, adapter)
    assert logits.shape == (x.shape[0], SEQ_LEN, VOCAB_SIZE)


def test_gauss_differs_from_full(adapter):
    x = torch.randint(0, VOCAB_SIZE - 1, (1, SEQ_LEN))
    full = adapter.logits(x)
    ws = InputNoiseGauss(sigma=1.0, seed=7)
    weak = ws(x, None, adapter)
    assert not torch.allclose(full, weak)


# ---------------------------------------------------------------------------
# Existing tests — LowNFEWeakSelf
# ---------------------------------------------------------------------------

def test_low_nfe_nfe1_output_shape(adapter):
    x = make_x()
    ws = LowNFEWeakSelf(nfe=1)
    logits = ws(x, None, adapter)
    assert logits.shape == (x.shape[0], SEQ_LEN, VOCAB_SIZE)


def test_low_nfe_nfe1_uses_fully_masked():
    """NFE=1 should produce logits from a fully-masked input."""
    adapter = SyntheticAdapter(seed=42)
    x = make_x(B=1)
    full_masked = torch.full_like(x, MASK_ID)
    expected = adapter.logits(full_masked)
    ws = LowNFEWeakSelf(nfe=1)
    got = ws(x, None, adapter)
    assert torch.allclose(expected, got)


def test_low_nfe_nfe3_output_shape(adapter):
    x = make_x()
    ws = LowNFEWeakSelf(nfe=3)
    logits = ws(x, None, adapter)
    assert logits.shape == (x.shape[0], SEQ_LEN, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Existing tests — ReducedExpertWeakSelf
# ---------------------------------------------------------------------------

def test_reduced_expert_raises():
    adapter = SyntheticAdapter(seed=42)
    x = make_x()
    ws = ReducedExpertWeakSelf(top_k=4)
    with pytest.raises(NotImplementedError):
        ws(x, None, adapter)


# ---------------------------------------------------------------------------
# New tests — LayerDropWeakSelf
# ---------------------------------------------------------------------------

def test_layer_drop_output_shape(adapter):
    """LayerDropWeakSelf returns [B, L, V] logits."""
    x = make_x()
    ws = LayerDropWeakSelf(k=1)
    logits = ws(x, None, adapter)
    assert logits.shape == (x.shape[0], SEQ_LEN, VOCAB_SIZE), (
        f"Expected ({x.shape[0]}, {SEQ_LEN}, {VOCAB_SIZE}), got {logits.shape}"
    )


def test_layer_drop_differs_from_full(adapter):
    """Dropping a layer should produce different logits from the full model."""
    x = make_x(B=1)
    full = adapter.logits(x)
    ws = LayerDropWeakSelf(k=1)
    weak = ws(x, None, adapter)
    assert not torch.allclose(full, weak), (
        "LayerDrop weak logits should differ from full-model logits"
    )


def test_layer_drop_restores_model_state(adapter):
    """After the weak pass, the full model returns identical logits as before."""
    x = make_x(B=1)
    before = adapter.logits(x).clone()
    ws = LayerDropWeakSelf(k=1)
    ws(x, None, adapter)
    after = adapter.logits(x)
    assert torch.allclose(before, after), (
        "Full model logits changed after LayerDrop — layers not restored"
    )


def test_layer_drop_k1_output_shape(adapter):
    """LayerDropWeakSelf with k=1 returns correct shape on the 2-layer synthetic model."""
    # SyntheticAdapter has N_LAYERS=2; k=1 drops 1 layer and keeps 1.
    # k=2 would raise NotImplementedError (would leave no layers).
    x = make_x()
    ws = LayerDropWeakSelf(k=1)
    logits = ws(x, None, adapter)
    assert logits.shape[0] == x.shape[0]
    assert logits.shape[1] == SEQ_LEN
    assert logits.shape[2] == VOCAB_SIZE


def test_layer_drop_restores_on_exception():
    """Decoder layers are restored when an exception occurs inside the weak pass."""
    class _BrokenAdapter(SyntheticAdapter):
        def logits(self, x_t, cond=None):
            raise RuntimeError("simulated error inside weak pass")

    broken = _BrokenAdapter(seed=42)
    x = make_x(B=1)
    ws = LayerDropWeakSelf(k=1)
    with pytest.raises(RuntimeError):
        ws(x, None, broken)
    # All 2 layers must be restored after the exception.
    assert len(broken._decoder_layers()) == 2, "Decoder layers not restored after exception"


# ---------------------------------------------------------------------------
# New tests — InferenceDropoutWeakSelf
# ---------------------------------------------------------------------------

def test_inference_dropout_output_shape(adapter):
    """InferenceDropoutWeakSelf returns [B, L, V] logits."""
    x = make_x()
    ws = InferenceDropoutWeakSelf(p=0.5, seed=42)
    logits = ws(x, None, adapter)
    assert logits.shape == (x.shape[0], SEQ_LEN, VOCAB_SIZE), (
        f"Expected ({x.shape[0]}, {SEQ_LEN}, {VOCAB_SIZE}), got {logits.shape}"
    )


def test_inference_dropout_differs_on_dropout_model():
    """Enabling dropout at inference changes logits on a model that uses nn.Dropout
    in its forward path.

    TransformerEncoderLayer eval-mode fastpaths bypass individual nn.Dropout modules,
    so we test the construction with a tiny standalone model whose forward explicitly
    calls an nn.Dropout module (as the InferenceDropoutWeakSelf implementation expects).
    """
    from autoguidance.models.base import ModelAdapter

    class _MLPWithDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(p=0.1)
            self.fc1 = nn.Linear(VOCAB_SIZE, 64, bias=False)
            self.fc2 = nn.Linear(64, VOCAB_SIZE, bias=False)
            with torch.no_grad():
                nn.init.normal_(self.fc1.weight, std=0.1)
                nn.init.normal_(self.fc2.weight, std=0.1)

        def forward(self, x):
            # x: [B, L] long → embed as float one-hot, apply dropout, project back
            h = self.fc1(nn.functional.one_hot(x, VOCAB_SIZE).float())
            h = self.drop(h)
            return self.fc2(h)

    class _DropoutAdapter(ModelAdapter):
        """Minimal adapter that wraps _MLPWithDropout for InferenceDropoutWeakSelf."""
        def __init__(self):
            self._model_obj = _MLPWithDropout().eval()
            self._device = torch.device("cpu")

        # _model attribute (not property) so InferenceDropoutWeakSelf can iterate modules
        @property
        def _model(self):
            return self._model_obj

        def logits(self, x_t, cond=None):
            with torch.no_grad():
                return self._model_obj(x_t)

        def embed(self, x_t):
            with torch.no_grad():
                return nn.functional.one_hot(x_t, VOCAB_SIZE).float() @ self._model_obj.fc1.weight.T

        def logits_from_embed(self, embeds, cond=None):
            with torch.no_grad():
                h = self._model_obj.drop(embeds)
                return self._model_obj.fc2(h)

        @property
        def mask_token_id(self): return MASK_ID

        @property
        def vocab_size(self): return VOCAB_SIZE

        @property
        def device(self): return self._device

    da = _DropoutAdapter()
    x = make_x(B=1)
    full = da.logits(x).clone()
    ws = InferenceDropoutWeakSelf(p=0.9, seed=7)   # high p ensures reliable difference
    weak = ws(x, None, da)
    assert not torch.allclose(full, weak), (
        "InferenceDropout should change logits when nn.Dropout is in the forward path"
    )


def test_inference_dropout_restores_eval_mode(adapter):
    """After the weak pass, the model is back in eval mode (training=False)."""
    assert not adapter._model.training, "Precondition: model starts in eval mode"
    x = make_x(B=1)
    ws = InferenceDropoutWeakSelf(p=0.5, seed=3)
    ws(x, None, adapter)
    assert not adapter._model.training, (
        "Model should be in eval mode after InferenceDropoutWeakSelf"
    )


def test_inference_dropout_restores_eval_on_exception():
    """Model eval mode is restored even if an exception is raised inside."""
    class _BrokenMid(SyntheticAdapter):
        def logits(self, x_t, cond=None):
            raise RuntimeError("boom")

    broken = _BrokenMid(seed=42)
    x = make_x(B=1)
    ws = InferenceDropoutWeakSelf(p=0.3, seed=5)
    with pytest.raises(RuntimeError):
        ws(x, None, broken)
    assert not broken._model.training, (
        "Model not restored to eval mode after exception in InferenceDropoutWeakSelf"
    )


def test_inference_dropout_full_logits_unchanged_after(adapter):
    """Full model logits are bit-identical before and after a weak pass."""
    x = make_x(B=1)
    before = adapter.logits(x).clone()
    ws = InferenceDropoutWeakSelf(p=0.5, seed=11)
    ws(x, None, adapter)
    after = adapter.logits(x)
    assert torch.allclose(before, after), (
        "Full model logits changed after InferenceDropoutWeakSelf — state not clean"
    )


# ---------------------------------------------------------------------------
# load_weak_self factory: layer_drop and inference_dropout
# ---------------------------------------------------------------------------

def test_load_weak_self_layer_drop():
    """load_weak_self('layer_drop', cfg) returns a LayerDropWeakSelf instance."""
    cfg = Phase0Config()
    ws = load_weak_self("layer_drop", cfg)
    assert isinstance(ws, LayerDropWeakSelf)


def test_load_weak_self_inference_dropout():
    """load_weak_self('inference_dropout', cfg) returns an InferenceDropoutWeakSelf."""
    cfg = Phase0Config()
    ws = load_weak_self("inference_dropout", cfg)
    assert isinstance(ws, InferenceDropoutWeakSelf)


def test_load_weak_self_layer_drop_uses_cfg_k():
    """load_weak_self passes cfg.layer_drop_k as k to LayerDropWeakSelf."""
    cfg = Phase0Config()
    cfg.layer_drop_k = 1   # use 1 to stay compatible with 2-layer SyntheticAdapter
    ws = load_weak_self("layer_drop", cfg)
    assert isinstance(ws, LayerDropWeakSelf)
    assert ws.k == cfg.layer_drop_k, (
        f"Expected ws.k={cfg.layer_drop_k}, got {ws.k}"
    )


def test_load_weak_self_inference_dropout_uses_cfg_dropout_p():
    """load_weak_self passes cfg.dropout_p as p to InferenceDropoutWeakSelf."""
    cfg = Phase0Config()
    cfg.dropout_p = 0.2
    ws = load_weak_self("inference_dropout", cfg)
    assert isinstance(ws, InferenceDropoutWeakSelf)
    assert ws.p == cfg.dropout_p, (
        f"Expected ws.p={cfg.dropout_p}, got {ws.p}"
    )

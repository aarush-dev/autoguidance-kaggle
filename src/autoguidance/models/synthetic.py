"""Tiny CPU masked-LM for fast unit + smoke tests. No GPU required.

Architecture: 2-layer transformer encoder, vocab=100, seq=32, d_model=64.
Token 99 is MASK. Everything else is a "word".

Optional MoE-sim mode (is_moe=True): attaches a trivial router module carrying
an integer `top_k` attribute so the reduced-expert weak self + moe_patch can be
exercised on CPU without a real mixture-of-experts model.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import LongTensor, BoolTensor, FloatTensor

from autoguidance.models.base import ModelAdapter

VOCAB_SIZE = 100
SEQ_LEN = 32
MASK_ID = 99
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DEFAULT_TOP_K = 8


class _MoERouter(nn.Module):
    """Trivial stand-in for an MoE router so reduced-expert patching is testable.

    Carries an integer `top_k` attribute that moe_patch can read/override.
    Has no effect on the forward pass — it exists purely to be discovered/patched.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
        super().__init__()
        self.top_k = int(top_k)


class _TinyMaskedLM(nn.Module):
    def __init__(self, seed: int = 42, is_moe: bool = False) -> None:
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        # Optional MoE router (only present in MoE-sim mode).
        self.router = _MoERouter() if is_moe else None
        # deterministic init
        with torch.no_grad():
            for p in self.parameters():
                nn.init.normal_(p, 0.0, 0.02)

    def forward_from_embeds(self, embeds: FloatTensor) -> FloatTensor:
        h = self.encoder(embeds)
        return self.head(h)

    def forward(self, input_ids: LongTensor) -> FloatTensor:
        return self.forward_from_embeds(self.embed(input_ids))


class SyntheticAdapter(ModelAdapter):
    """ModelAdapter wrapping _TinyMaskedLM. CPU-only, instant load."""

    def __init__(self, cfg=None, seed: int = 42, is_moe: bool = False) -> None:
        if cfg is not None and hasattr(cfg, "seed"):
            seed = cfg.seed
        self._is_moe = bool(is_moe)
        self._model = _TinyMaskedLM(seed=seed, is_moe=self._is_moe).eval()
        self._device = torch.device("cpu")

    # ---- ModelAdapter interface ----

    def logits(self, x_t: LongTensor, cond: Optional[dict] = None) -> FloatTensor:
        with torch.no_grad():
            return self._model(x_t.to(self._device))

    def embed(self, x_t: LongTensor) -> FloatTensor:
        with torch.no_grad():
            return self._model.embed(x_t.to(self._device))

    def logits_from_embed(
        self, embeds: FloatTensor, cond: Optional[dict] = None
    ) -> FloatTensor:
        with torch.no_grad():
            return self._model.forward_from_embeds(embeds.to(self._device))

    @property
    def mask_token_id(self) -> int:
        return MASK_ID

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def device(self) -> torch.device:
        return self._device

    def encode(self, text: str) -> LongTensor:
        # Simple whitespace tokenizer: each word → hash mod (VOCAB_SIZE - 1)
        tokens = [hash(w) % (VOCAB_SIZE - 1) for w in text.split()]
        tokens = tokens[:SEQ_LEN]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    def decode(self, token_ids: LongTensor) -> str:
        ids = token_ids.squeeze(0).tolist()
        return " ".join(str(i) for i in ids if i != MASK_ID)

    # ---- noise process / capability hooks ----

    def corrupt_positions(
        self,
        ids: LongTensor,
        mask: BoolTensor,
        generator: Optional[torch.Generator] = None,
    ) -> LongTensor:
        out = ids.clone()
        out[mask] = MASK_ID
        return out

    @property
    def supports_embed_noise(self) -> bool:
        return True

    @property
    def is_moe(self) -> bool:
        return self._is_moe

    def _decoder_layers(self) -> nn.ModuleList:
        # nn.TransformerEncoder stores its stacked layers in .layers (a ModuleList).
        return self._model.encoder.layers

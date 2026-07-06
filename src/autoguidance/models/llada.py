"""LLaDA-8B-Instruct adapter — single full-bf16 / single-GPU load path.

Loads GSAI-ML/LLaDA-8B-Instruct in bf16 onto one CUDA device (default cuda:0).
No bitsandbytes, no 4/8-bit, no fp16_offload, no dual-GPU split, no Pascal patch:
the RTX PRO 6000 (96 GB) holds the ~16 GB bf16 model with room to spare.

Offline-first: when a local directory is provided (Kaggle dataset mount), it is
loaded with local_files_only=True; otherwise the hub MODEL_ID is used.

API confirmed from:
  https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
  https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/main/modeling_llada.py
  cached modeling_llada.py: model.model.transformer.blocks is the nn.ModuleList
  of 32 LLaDABlock layers (config.block_group_size == 1).
"""
from __future__ import annotations
import os
from typing import Optional

import torch
import torch.nn as nn
from torch import LongTensor, FloatTensor

from autoguidance.models.base import ModelAdapter

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
_MASK_ID = 126336   # confirmed from config.json + generate.py
_EOS_ID = 126081    # confirmed from config.json


def _resolve_source(cfg, model_path: Optional[str]) -> tuple:
    """Return (source, local) where source is a local dir or the hub id."""
    path = (
        model_path
        or getattr(cfg, "llada_path", None)
        or getattr(cfg, "model_path", None)
    )
    if path is not None and os.path.isdir(path):
        return path, True
    return MODEL_ID, False


def _load_model(source: str, local: bool, device_main: str):
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PreTrainedModel

    # LLaDA's remote modeling was written for transformers ~4.46. Under
    # transformers 5.x, from_pretrained expects APIs the old class lacks:
    #   * every model exposes `all_tied_weights_keys`  (5.x)
    #   * tie_weights is called as `tie_weights(missing_keys=...)`  (5.x)
    #   * only eager attention is implemented by the remote code
    # We make loading work under BOTH 4.46.x and 5.x so a mismatched runtime
    # transformers version doesn't hard-fail. LLaDA-8B-Base does not tie its
    # input/output embeddings (separate ff_out in the checkpoint), so an empty
    # tied-keys mapping and a kwargs-tolerant tie_weights are both correct.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    common = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device_main},
        attn_implementation="eager",   # remote modeling implements eager only
        local_files_only=local,
    )

    model = None
    try:
        # Preload the remote model class WITHOUT building it, patch tie_weights to
        # swallow the transformers-5.x `missing_keys` (and any other) kwarg, then
        # build. AutoModel.from_pretrained would tie mid-load before we can patch.
        cfg = AutoConfig.from_pretrained(
            source, trust_remote_code=True, local_files_only=local
        )
        amap = getattr(cfg, "auto_map", None) or {}
        ref = amap.get("AutoModel") or amap.get("AutoModelForCausalLM")
        if ref:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            ModelClass = get_class_from_dynamic_module(ref, source, local_files_only=local)
            if not getattr(ModelClass, "_llada_tie_patched", False):
                _orig_tie = ModelClass.tie_weights
                def _tie_tolerant(self, *args, **kwargs):   # drop 5.x-only kwargs
                    return _orig_tie(self)
                ModelClass.tie_weights = _tie_tolerant
                ModelClass._llada_tie_patched = True
            model = ModelClass.from_pretrained(source, config=cfg, **common).eval()
    except Exception as e:
        print(f"[LLaDAAdapter] remote-class tie_weights patch skipped ({type(e).__name__}: {e})")

    if model is None:
        model = AutoModel.from_pretrained(source, **common).eval()

    # LLaDAConfig doesn't set the transformers-standard config flags the runtime
    # reads during forward (LLaDA is a diffusion LM — no KV cache). Fill safe
    # defaults so `config.use_cache` & friends don't AttributeError mid-forward.
    for _attr, _default in (("use_cache", False), ("output_attentions", False),
                            ("output_hidden_states", False)):
        if not hasattr(model.config, _attr):
            setattr(model.config, _attr, _default)

    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=True, local_files_only=local
    )
    return model, tokenizer


def _find_decoder_layers(model) -> tuple:
    """Locate the transformer block ModuleList. Returns (attr_path, module_list)."""
    inner = getattr(model, "model", model)
    transformer = getattr(inner, "transformer", None)
    if transformer is not None:
        blocks = getattr(transformer, "blocks", None)
        if isinstance(blocks, nn.ModuleList):
            return "model.transformer.blocks", blocks
    # Fallback: scan for the longest ModuleList (the decoder stack).
    best_path, best = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if best is None or len(mod) > len(best):
                best_path, best = name, mod
    if best is None:
        raise RuntimeError("LLaDAAdapter: could not locate decoder layer ModuleList")
    return best_path, best


class LLaDAAdapter(ModelAdapter):
    """ModelAdapter for LLaDA-8B-Instruct. All logit calls are no-grad.

    Full model and weak self share this single bf16 adapter so comparisons are
    never contaminated by precision mismatch.
    """

    def __init__(self, cfg, model_path: Optional[str] = None) -> None:
        device_main = getattr(cfg, "device_main", "cuda:0")
        source, local = _resolve_source(cfg, model_path)
        print(f"[LLaDAAdapter] Loading {source!r} (local={local}) "
              f"@ bf16 on {device_main} (attn=eager, device_map={{'':{device_main!r}}}) …")
        self._model, self._tokenizer = _load_model(source, local, device_main)
        self._device = torch.device(device_main)
        # Resolve + cache the decoder layer stack for layer-drop.
        self._layers_path, self._layers = _find_decoder_layers(self._model)
        print(f"[LLaDAAdapter] device={self._device} dtype=bfloat16 "
              f"vocab={self.vocab_size} n_layers={len(self._layers)} "
              f"decoder_layers='{self._layers_path}'")

    # ---- ModelAdapter interface ----

    @torch.no_grad()
    def logits(self, x_t: LongTensor, cond: Optional[dict] = None) -> FloatTensor:
        """Single forward pass. Returns [B, L, vocab] logits for ALL positions."""
        x = x_t.to(self._device)
        # Attention mask: 1 for every real token (MASK tokens included), 0 only
        # for padding. LLaDA's masking lives in the token IDs, not the attn mask.
        attn_mask = (x != _EOS_ID).long()
        return self._model(input_ids=x, attention_mask=attn_mask).logits

    @torch.no_grad()
    def embed(self, x_t: LongTensor) -> FloatTensor:
        """Return input embeddings [B, L, d_model]."""
        embed_layer = self._model.get_input_embeddings()
        return embed_layer(x_t.to(self._device))

    @torch.no_grad()
    def logits_from_embed(
        self, embeds: FloatTensor, cond: Optional[dict] = None
    ) -> FloatTensor:
        """Forward pass from embeddings [B, L, d_model] → logits [B, L, vocab].

        Used by the Gaussian-embedding weak self.
        """
        return self._model(inputs_embeds=embeds.to(self._device)).logits

    @property
    def mask_token_id(self) -> int:
        return _MASK_ID

    @property
    def vocab_size(self) -> int:
        return self._model.config.vocab_size

    @property
    def device(self) -> torch.device:
        return self._device

    def encode(self, text: str, max_length: int = 256) -> LongTensor:
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return enc["input_ids"].to(self._device)

    def decode(self, token_ids: LongTensor) -> str:
        ids = token_ids.squeeze(0)
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    # ---- noise process / capability hooks ----

    def corrupt_positions(self, ids, mask, generator=None):
        """LLaDA forward process is absorbing-state masking: set to MASK_ID."""
        out = ids.clone()
        out[mask] = _MASK_ID
        return out

    @property
    def supports_embed_noise(self) -> bool:
        return True

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    def _decoder_layers(self) -> nn.ModuleList:
        return self._layers

    # ---- CFG helpers (used by samplers) ----

    @torch.no_grad()
    def logits_cfg(
        self,
        x_t: LongTensor,
        prompt_mask: LongTensor,
        cfg_scale: float,
    ) -> FloatTensor:
        """One forward pass returning CFG-guided logits.

        Official LLaDA CFG via batch concat:
            un_x = x_t with prompt positions set to MASK
            x_ = [x_t; un_x]
            logits, un_logits = chunk(model(x_))
            guided = un_logits + (cfg_scale + 1) * (logits - un_logits)
        """
        x = x_t.to(self._device)
        un_x = x.clone()
        un_x[prompt_mask.to(self._device)] = _MASK_ID
        x_ = torch.cat([x, un_x], dim=0)           # [2B, L]
        attn = (x_ != _EOS_ID).long()
        out = self._model(input_ids=x_, attention_mask=attn).logits  # [2B, L, V]
        logits, un_logits = torch.chunk(out, 2, dim=0)
        return un_logits + (cfg_scale + 1) * (logits - un_logits)

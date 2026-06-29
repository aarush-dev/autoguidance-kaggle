"""DiffusionGemma adapter — google/diffusiongemma-26B-A4B-it (MoE).

Full bf16 on cuda:0 (single RTX PRO 6000, 96 GB). NO bitsandbytes / 4bit / 8bit /
dual-GPU / CPU offload. 26B total / ~A4B active MoE.

load_diffusiongemma DEBUG NOTE
------------------------------
The exact HuggingFace load API for this model is UNVERIFIED at build time, so
this adapter is deliberately defensive:

  * It resolves a source: a local dir (cfg.diffusiongemma_path / .diffusiongemma_dir
    / .model_path) if provided, else the hub id MODEL_ID.
  * It loads AutoConfig FIRST and PRINTS the resolved config class, model_type,
    architectures and num_hidden_layers so a human can confirm the real shape.
  * It then tries, in order:
        1. AutoModelForCausalLM (native, no remote code)
        2. AutoModel            (native, no remote code)
        3. AutoModelForCausalLM (trust_remote_code=True)
        4. AutoModel            (trust_remote_code=True)
    PRINTING the actual loaded class on success and the exception type on each
    failure. If all fail it re-raises the last error with the full attempt log.

  * logits() returns per-position logits [B, L, vocab] REGARDLESS of the model's
    internal canvas / encoder-decoder shape: for encoder-decoder configs x_t is
    fed as decoder_input_ids; the output is cropped/padded to match L.

  * This model is MoE (is_moe=True), defines no guaranteed mask token
    (mask_token_id may raise), and embedding-space noise is not supported
    (supports_embed_noise=False → the Gauss weak-self auto-skips). corrupt_positions
    therefore uses random-token replacement from U(0, vocab).

If the resolved class/config printed at build time disagrees with the assumptions
here, adjust _decoder_layers()/logits() — the resolved-path prints make that easy.
"""
from __future__ import annotations
from typing import Optional, List

import torch
from torch import LongTensor, FloatTensor

from autoguidance.models.base import ModelAdapter

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

# Candidate dotted attribute paths to the decoder transformer-layer ModuleList.
# Tried in order; first one that resolves to a non-empty nn.ModuleList wins.
_LAYER_PATHS = (
    "model.model.layers",
    "model.layers",
    "model.model.decoder.layers",
    "model.decoder.layers",
    "model.model.text_model.layers",
    "model.model.language_model.layers",
    "model.language_model.model.layers",
)


def _resolve_source(cfg) -> str:
    """Local dir override (if present) else the hub id."""
    for attr in ("diffusiongemma_path", "diffusiongemma_dir", "model_path", "diffusiongemma_local_dir"):
        p = getattr(cfg, attr, None)
        if p:
            print(f"[DiffusionGemma] using local source from cfg.{attr}: {p}")
            return str(p)
    print(f"[DiffusionGemma] no local-dir cfg attr found; using hub id: {MODEL_ID}")
    return MODEL_ID


def _num_layers_from_config(config) -> Optional[int]:
    """num_hidden_layers, handling text_config / language_model nesting."""
    for c in (config,
              getattr(config, "text_config", None),
              getattr(config, "language_model_config", None)):
        if c is not None and getattr(c, "num_hidden_layers", None) is not None:
            return int(c.num_hidden_layers)
    return None


def _load_model(source: str, device_main: str):
    """Defensive load. PRINTS the real config + the class that actually loaded."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    print(f"[DiffusionGemma] ===== load_diffusiongemma: source={source} device={device_main} dtype=bf16 =====")

    config = None
    # AutoConfig first (try native then remote-code) so we can print the real shape.
    for trc in (False, True):
        try:
            config = AutoConfig.from_pretrained(source, trust_remote_code=trc)
            print(f"[DiffusionGemma] AutoConfig OK (trust_remote_code={trc}): "
                  f"class={type(config).__name__} model_type={getattr(config,'model_type',None)} "
                  f"architectures={getattr(config,'architectures',None)} "
                  f"is_encoder_decoder={getattr(config,'is_encoder_decoder',False)} "
                  f"num_hidden_layers={_num_layers_from_config(config)} "
                  f"vocab_size={getattr(config,'vocab_size',getattr(getattr(config,'text_config',None),'vocab_size',None))}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[DiffusionGemma] AutoConfig FAILED (trust_remote_code={trc}): "
                  f"{type(e).__name__}: {e}")

    common = dict(torch_dtype=torch.bfloat16,
                  device_map={"": device_main},
                  low_cpu_mem_usage=True)

    attempts = [
        ("AutoModelForCausalLM(native)",            AutoModelForCausalLM, False),
        ("AutoModel(native)",                       AutoModel,            False),
        ("AutoModelForCausalLM(trust_remote_code)", AutoModelForCausalLM, True),
        ("AutoModel(trust_remote_code)",            AutoModel,            True),
    ]

    last_err = None
    for label, klass, trc in attempts:
        try:
            print(f"[DiffusionGemma] trying {label} …")
            model = klass.from_pretrained(source, trust_remote_code=trc, **common).eval()
            print(f"[DiffusionGemma] LOADED via {label} → class={type(model).__name__}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[DiffusionGemma] {label} FAILED: {type(e).__name__}: {e}")
            last_err = e
            model = None
    if model is None:
        raise RuntimeError(
            f"[DiffusionGemma] all load attempts failed for source={source!r}. "
            f"Last error: {type(last_err).__name__}: {last_err}"
        ) from last_err

    # Tokenizer (best-effort; native then remote-code).
    tokenizer = None
    for trc in (False, True):
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trc)
            print(f"[DiffusionGemma] tokenizer OK (trust_remote_code={trc}): {type(tokenizer).__name__}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[DiffusionGemma] tokenizer FAILED (trust_remote_code={trc}): {type(e).__name__}: {e}")

    return model, tokenizer, model.config


class DiffusionGemmaAdapter(ModelAdapter):
    """ModelAdapter for DiffusionGemma 26B-A4B (MoE). bf16 / cuda:0, no-grad inference."""

    def __init__(self, cfg) -> None:
        device_main = getattr(cfg, "device_main", "cuda:0")
        precision = getattr(cfg, "precision", "bf16")
        if precision not in ("bf16", "bfloat16"):
            print(f"[DiffusionGemma] WARNING: precision={precision!r}; this adapter only "
                  f"supports bf16 on this hardware — forcing bf16.")
        source = _resolve_source(cfg)
        self._model, self._tokenizer, self._config = _load_model(source, device_main)
        self._device = torch.device(device_main)
        self._is_enc_dec = bool(getattr(self._config, "is_encoder_decoder", False))

        # vocab size (handle text_config nesting)
        v = getattr(self._config, "vocab_size", None)
        if v is None and getattr(self._config, "text_config", None) is not None:
            v = getattr(self._config.text_config, "vocab_size", None)
        if v is None:
            v = int(self._model.get_input_embeddings().num_embeddings)
        self._vocab_size = int(v)

        # mask token — only if the model/tokenizer actually defines one
        mid = None
        if self._tokenizer is not None and getattr(self._tokenizer, "mask_token_id", None) is not None:
            mid = int(self._tokenizer.mask_token_id)
        elif getattr(self._config, "mask_token_id", None) is not None:
            mid = int(self._config.mask_token_id)
        self._mask_token_id = mid

        self._n_layers = _num_layers_from_config(self._config)

        print(f"[DiffusionGemma] ready: device={self._device} dtype=bf16 "
              f"vocab_size={self._vocab_size} n_layers={self._n_layers} "
              f"is_moe={self.is_moe} is_encoder_decoder={self._is_enc_dec} "
              f"mask_token_id={self._mask_token_id} "
              f"supports_embed_noise={self.supports_embed_noise}")

    # ---- internal helpers ----

    def _align(self, logits: FloatTensor, target_len: int) -> FloatTensor:
        """Crop/pad logits seq dim to exactly target_len → [B, target_len, V]."""
        L = logits.shape[1]
        if L == target_len:
            return logits
        if L > target_len:
            print(f"[DiffusionGemma] cropping logits seq {L} → {target_len}")
            return logits[:, :target_len, :]
        print(f"[DiffusionGemma] padding logits seq {L} → {target_len} (repeat last position)")
        pad = logits[:, -1:, :].expand(-1, target_len - L, -1)
        return torch.cat([logits, pad], dim=1)

    # ---- ModelAdapter interface ----

    @torch.no_grad()
    def logits(self, x_t: LongTensor, cond: Optional[dict] = None) -> FloatTensor:
        """Per-position logits [B, L, vocab] for ALL positions of x_t."""
        x = x_t.to(self._device)
        B, L = x.shape
        attn = torch.ones_like(x)
        if self._is_enc_dec:
            # encoder-decoder: x_t is the (masked) canvas → decoder_input_ids.
            enc_ids = None
            if cond is not None and cond.get("input_ids", None) is not None:
                enc_ids = cond["input_ids"].to(self._device)
            if enc_ids is None:
                enc_ids = x  # no separate prompt; reuse canvas as encoder input
            out = self._model(input_ids=enc_ids, decoder_input_ids=x)
        else:
            out = self._model(input_ids=x, attention_mask=attn)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return self._align(logits, L)

    @torch.no_grad()
    def embed(self, x_t: LongTensor) -> FloatTensor:
        """Input embeddings [B, L, d_model]."""
        return self._model.get_input_embeddings()(x_t.to(self._device))

    @torch.no_grad()
    def logits_from_embed(self, embeds: FloatTensor, cond: Optional[dict] = None) -> FloatTensor:
        """Best-effort forward from embeddings. Gauss weak-self is gated off for this
        model (supports_embed_noise=False), so this is provided only for completeness."""
        e = embeds.to(self._device)
        L = e.shape[1]
        if self._is_enc_dec:
            out = self._model(decoder_inputs_embeds=e)
        else:
            out = self._model(inputs_embeds=e)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return self._align(logits, L)

    # ---- properties ----

    @property
    def mask_token_id(self) -> int:
        if self._mask_token_id is None:
            raise NotImplementedError(
                "[DiffusionGemma] model defines no mask token; corruption uses "
                "random-token replacement via corrupt_positions()."
            )
        return self._mask_token_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_moe(self) -> bool:
        return True

    @property
    def supports_embed_noise(self) -> bool:
        return False

    @property
    def n_layers(self) -> int:
        if self._n_layers is None:
            raise NotImplementedError("[DiffusionGemma] num_hidden_layers not resolvable from config.")
        return self._n_layers

    # ---- noise process ----

    @torch.no_grad()
    def corrupt_positions(self, ids: LongTensor, mask: torch.BoolTensor, generator=None) -> LongTensor:
        """Replace masked positions with random tokens drawn from U(0, vocab).

        DiffusionGemma has no guaranteed [MASK] token, so the model's corruption
        process is random-token replacement (not remasking).
        """
        out = ids.clone()
        n = int(mask.sum().item())
        if n > 0:
            rand = torch.randint(0, self._vocab_size, (n,), generator=generator, dtype=out.dtype)
            out[mask] = rand.to(out.device)
        return out

    # ---- layer-drop hook ----

    def _decoder_layers(self):
        """Best-effort resolve the decoder transformer-layer nn.ModuleList.

        Tries known dotted paths first, then falls back to scanning all modules
        for the largest nn.ModuleList. PRINTS the resolved path.
        """
        import torch.nn as nn

        for path in _LAYER_PATHS:
            obj = self._model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
                print(f"[DiffusionGemma] _decoder_layers resolved path: {path} (len={len(obj)})")
                return obj

        # Fallback: largest ModuleList in the model.
        best, best_name, best_len = None, None, -1
        for name, mod in self._model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
                best, best_name, best_len = mod, name, len(mod)
        if best is None:
            raise NotImplementedError("[DiffusionGemma] could not resolve a decoder layer ModuleList.")
        print(f"[DiffusionGemma] _decoder_layers fallback path: {best_name} (len={best_len})")
        return best

    # ---- tokenizer helpers ----

    def encode(self, text: str, max_length: int = 256) -> LongTensor:
        if self._tokenizer is None:
            raise NotImplementedError("[DiffusionGemma] no tokenizer loaded.")
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        return enc["input_ids"].to(self._device)

    def decode(self, token_ids: LongTensor) -> str:
        if self._tokenizer is None:
            raise NotImplementedError("[DiffusionGemma] no tokenizer loaded.")
        return self._tokenizer.decode(token_ids.squeeze(0), skip_special_tokens=True)

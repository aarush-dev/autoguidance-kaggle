from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from torch import LongTensor, BoolTensor, FloatTensor


class ModelAdapter(ABC):
    """Uniform interface over all masked-diffusion LMs.

    Implementations: SyntheticAdapter, LLaDAAdapter (stub: DreamAdapter, DiffusionGemmaAdapter).
    All returned tensors are on the adapter's configured device.
    """

    # ---- required ----

    @abstractmethod
    def logits(self, x_t: LongTensor, cond: Optional[dict] = None) -> FloatTensor:
        """Single forward pass.

        Args:
            x_t: [batch, seq] token IDs; masked positions contain mask_token_id.
            cond: Optional conditioning dict (e.g. {"input_ids": prompt_ids}).
        Returns:
            logits: [batch, seq, vocab] float32.
        """

    @abstractmethod
    def embed(self, x_t: LongTensor) -> FloatTensor:
        """Return input embeddings before the transformer stack.

        Returns: [batch, seq, d_model] float32.
        """

    @abstractmethod
    def logits_from_embed(
        self, embeds: FloatTensor, cond: Optional[dict] = None
    ) -> FloatTensor:
        """Forward pass starting from pre-computed embeddings.

        Needed for Construction A (Gaussian noise on embeddings).
        Returns: [batch, seq, vocab] float32.
        """

    # ---- properties ----

    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        """Token ID used for masked positions."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Primary compute device for this adapter."""

    # ---- optional helpers with defaults ----

    def encode(self, text: str) -> LongTensor:
        """Tokenize text → [1, seq] LongTensor on adapter device."""
        raise NotImplementedError

    def decode(self, token_ids: LongTensor) -> str:
        """Decode [seq] or [1, seq] → string."""
        raise NotImplementedError

    # ---- noise process / capability hooks (weak-self constructions) ----

    def corrupt_positions(
        self,
        ids: LongTensor,
        mask: BoolTensor,
        generator: Optional[torch.Generator] = None,
    ) -> LongTensor:
        """Replace the masked positions per this model's forward noise process.

        Args:
            ids: [batch, seq] token IDs (a clean / partially-decoded sequence).
            mask: [batch, seq] bool, True where positions should be corrupted.
            generator: optional RNG for stochastic noise processes.
        Returns:
            new ids tensor (same shape) with `mask` positions corrupted. The base
            implementation applies absorbing-state masking (set to mask_token_id),
            which is correct for LLaDA-style masked diffusion. Models with a
            different forward process override this.
        """
        out = ids.clone()
        out[mask] = self.mask_token_id
        return out

    @property
    def supports_embed_noise(self) -> bool:
        """Whether logits_from_embed is usable for Gaussian-embedding weak selves."""
        return True

    @property
    def n_layers(self) -> int:
        """Number of stacked transformer decoder layers (for layer-drop)."""
        return len(self._decoder_layers())

    @property
    def is_moe(self) -> bool:
        """Whether this model uses mixture-of-experts routing (for reduced-expert)."""
        return False

    def _decoder_layers(self) -> "nn.ModuleList":
        """Return the transformer block ModuleList (for layer-drop / hooks).

        Implementations must return the live nn.ModuleList so callers can
        temporarily mutate it. Raises if the model exposes no such stack.
        """
        raise NotImplementedError

"""Base sampler and shared masked-diffusion sampling loop."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import torch
from torch import LongTensor, FloatTensor

from autoguidance.models.base import ModelAdapter


@dataclass
class SamplerConfig:
    nfe: int = 128
    cfg_weight: float = 2.0
    acfg_threshold: float = 0.3      # for A-CFG: confidence below this → re-mask
    temperature: float = 0.0         # 0 = greedy
    seed: int = 42
    max_gen_len: int = 128


class SamplerBase(ABC):
    def __init__(self, adapter: ModelAdapter, cfg: SamplerConfig) -> None:
        self.adapter = adapter
        self.cfg = cfg

    @abstractmethod
    def sample(self, prompts: List[str], nfe: int) -> List[str]:
        """Generate text for each prompt using `nfe` denoising steps."""


def masked_diffusion_step(
    x: LongTensor,                          # [B, L] current sequence
    mask_id: int,
    get_logits: Callable[[LongTensor], FloatTensor],
    n_to_unmask_per_seq: LongTensor,        # [B] how many to unmask this step
    temperature: float,
    rng: torch.Generator,
    acfg_threshold: Optional[float] = None,
) -> LongTensor:
    """One masked-diffusion denoising step.

    Args:
        x: [B, L] current tokens (masked positions contain mask_id).
        mask_id: the MASK token ID.
        get_logits: callable that returns [B, L, vocab] logits.
        n_to_unmask_per_seq: [B] how many positions to permanently unmask this step.
        temperature: sampling temperature (0 = greedy argmax).
        rng: torch.Generator for reproducibility.
        acfg_threshold: if set, re-mask tokens below this confidence after unmasking.

    Returns:
        x updated in-place and returned.
    """
    B, L = x.shape
    is_masked = (x == mask_id)             # [B, L] bool

    logits = get_logits(x)                 # [B, L, vocab]
    probs = torch.softmax(logits.float(), dim=-1)  # [B, L, vocab]

    # Sample tokens at all positions
    if temperature > 0:
        flat_probs = probs.view(-1, probs.shape[-1])
        sampled = torch.multinomial(flat_probs, 1, generator=rng).view(B, L)
    else:
        sampled = logits.argmax(dim=-1)    # [B, L]

    # Confidence at each position
    confidence = probs.max(dim=-1).values  # [B, L]
    confidence[~is_masked] = float("inf")  # never re-select non-masked positions

    # Unmask the most confident masked positions
    for b in range(B):
        n = n_to_unmask_per_seq[b].item()
        if n <= 0:
            continue
        masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
        if len(masked_pos) == 0:
            continue
        n = min(n, len(masked_pos))
        conf_at_masked = confidence[b, masked_pos]
        top_idx = conf_at_masked.topk(n).indices     # highest confidence
        x[b, masked_pos[top_idx]] = sampled[b, masked_pos[top_idx]]

    # A-CFG: re-mask any now-unmasked tokens below confidence threshold
    if acfg_threshold is not None:
        # Re-check confidence using updated x (could re-run forward, but use current probs)
        # Positions that were just unmasked and have low confidence
        newly_unmasked = (~is_masked) | (x != mask_id)  # all unmasked positions
        low_conf = (confidence < acfg_threshold) & (x != mask_id)
        x[low_conf] = mask_id

    return x


def run_denoising_loop(
    adapter: ModelAdapter,
    input_ids: LongTensor,          # [B, prompt_len + gen_len]
    nfe: int,
    get_logits: Callable[[LongTensor], FloatTensor],
    temperature: float,
    seed: int,
    acfg_threshold: Optional[float] = None,
) -> LongTensor:
    """Full masked-diffusion denoising loop.

    Args:
        adapter: model adapter (provides mask_token_id, device).
        input_ids: starting sequence with prompt tokens + MASK tokens for generation.
        nfe: number of denoising steps.
        get_logits: callable returning [B, L, vocab] (allows CFG injection).
        temperature: sampling temperature.
        seed: random seed.
        acfg_threshold: if set, A-CFG re-masking threshold.

    Returns:
        [B, L] final token IDs.
    """
    mask_id = adapter.mask_token_id
    x = input_ids.clone().to(adapter.device)
    rng = torch.Generator(device=x.device)
    rng.manual_seed(seed)

    for step in range(nfe):
        is_masked = (x == mask_id)
        if not is_masked.any():
            break

        n_masked = is_masked.sum(dim=-1)           # [B]
        n_remaining = nfe - step
        # Linear schedule: ceil(n_masked / n_remaining_steps)
        n_to_unmask = ((n_masked.float() / n_remaining).ceil().long()).clamp(min=0)

        x = masked_diffusion_step(
            x=x,
            mask_id=mask_id,
            get_logits=get_logits,
            n_to_unmask_per_seq=n_to_unmask,
            temperature=temperature,
            rng=rng,
            acfg_threshold=acfg_threshold,
        )

    return x

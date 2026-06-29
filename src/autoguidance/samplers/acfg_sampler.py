"""A-CFG (Adaptive CFG) sampler for Phase 1 baseline.

Implements the A-CFG method from arXiv:2505.20199:
  At each denoising step, tokens that have been unmasked but have confidence
  below threshold τ are re-masked, allowing the model to reconsider them.
  This builds the unconditional reference adaptively instead of using a fixed empty prompt.

The re-masking is handled by passing acfg_threshold to the shared denoising loop.
CFG guidance is still applied on top (the paper combines both).
"""
from __future__ import annotations
from typing import List

import torch

from autoguidance.models.base import ModelAdapter
from autoguidance.samplers.base import SamplerBase, SamplerConfig, run_denoising_loop


class ACFGSampler(SamplerBase):
    """Adaptive CFG sampler (re-mask low-confidence tokens at each step)."""

    def __init__(self, adapter: ModelAdapter, cfg: SamplerConfig) -> None:
        super().__init__(adapter, cfg)

    def sample(self, prompts: List[str], nfe: int) -> List[str]:
        mask_id = self.adapter.mask_token_id
        device = self.adapter.device
        gen_len = self.cfg.max_gen_len
        threshold = self.cfg.acfg_threshold

        prompt_ids_list = [self.adapter.encode(p) for p in prompts]
        prompt_len = max(ids.shape[1] for ids in prompt_ids_list)
        gen_mask = torch.full((len(prompts), gen_len), mask_id, dtype=torch.long, device=device)

        padded = torch.full((len(prompts), prompt_len), mask_id, dtype=torch.long, device=device)
        for i, ids in enumerate(prompt_ids_list):
            L = ids.shape[1]
            padded[i, :L] = ids[0]

        input_ids = torch.cat([padded, gen_mask], dim=1)

        with torch.no_grad():
            output_ids = run_denoising_loop(
                adapter=self.adapter,
                input_ids=input_ids,
                nfe=nfe,
                get_logits=lambda x: self.adapter.logits(x),
                temperature=self.cfg.temperature,
                seed=self.cfg.seed,
                acfg_threshold=threshold,
            )

        generated = output_ids[:, prompt_len:]
        return [self.adapter.decode(generated[i]) for i in range(len(prompts))]

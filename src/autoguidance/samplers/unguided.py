"""Unguided masked-diffusion sampler (baseline)."""
from __future__ import annotations
from typing import List

import torch

from autoguidance.models.base import ModelAdapter
from autoguidance.samplers.base import SamplerBase, SamplerConfig, run_denoising_loop


class UnguidedSampler(SamplerBase):
    """Plain masked-diffusion decoding with no guidance."""

    def __init__(self, adapter: ModelAdapter, cfg: SamplerConfig) -> None:
        super().__init__(adapter, cfg)

    def sample(self, prompts: List[str], nfe: int) -> List[str]:
        """Generate continuations for each prompt using nfe denoising steps."""
        mask_id = self.adapter.mask_token_id
        device = self.adapter.device
        gen_len = self.cfg.max_gen_len

        # Tokenize prompts and append masked generation block
        prompt_ids_list = [self.adapter.encode(p) for p in prompts]
        prompt_len = max(ids.shape[1] for ids in prompt_ids_list)
        gen_mask = torch.full((len(prompts), gen_len), mask_id, dtype=torch.long, device=device)

        # Pad prompts to same length
        padded = torch.full((len(prompts), prompt_len), self.adapter.mask_token_id, dtype=torch.long, device=device)
        for i, ids in enumerate(prompt_ids_list):
            L = ids.shape[1]
            padded[i, :L] = ids[0]

        input_ids = torch.cat([padded, gen_mask], dim=1)   # [B, prompt_len + gen_len]

        with torch.no_grad():
            output_ids = run_denoising_loop(
                adapter=self.adapter,
                input_ids=input_ids,
                nfe=nfe,
                get_logits=lambda x: self.adapter.logits(x),
                temperature=self.cfg.temperature,
                seed=self.cfg.seed,
            )

        # Decode only the generated portion
        generated = output_ids[:, prompt_len:]
        return [self.adapter.decode(generated[i]) for i in range(len(prompts))]

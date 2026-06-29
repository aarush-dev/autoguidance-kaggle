"""CFG (Classifier-Free Guidance) sampler for Phase 1 baseline.

Implements the official LLaDA CFG formula (from generate.py):
    un_x = x with prompt positions replaced by MASK
    x_ = concat([x, un_x])   ← doubled batch
    logits, un_logits = split(model(x_))
    guided = un_logits + (cfg_scale + 1) * (logits - un_logits)
"""
from __future__ import annotations
from typing import List

import torch

from autoguidance.models.base import ModelAdapter
from autoguidance.samplers.base import SamplerBase, SamplerConfig, run_denoising_loop


class CFGSampler(SamplerBase):
    """Built-in CFG baseline using the LLaDA formula."""

    def __init__(self, adapter: ModelAdapter, cfg: SamplerConfig) -> None:
        super().__init__(adapter, cfg)

    def _cfg_logits(
        self,
        x: "torch.LongTensor",
        prompt_mask: "torch.BoolTensor",
        cfg_scale: float,
    ) -> "torch.FloatTensor":
        """Compute CFG logits via batch concat (no extra model call)."""
        mask_id = self.adapter.mask_token_id

        # Use adapter's built-in CFG helper if available (LLaDA has it)
        if hasattr(self.adapter, "logits_cfg"):
            return self.adapter.logits_cfg(x, prompt_mask, cfg_scale)

        # Generic fallback: two separate forward passes
        un_x = x.clone()
        un_x[prompt_mask] = mask_id
        logits = self.adapter.logits(x)
        un_logits = self.adapter.logits(un_x)
        return un_logits + (cfg_scale + 1) * (logits - un_logits)

    def sample(self, prompts: List[str], nfe: int) -> List[str]:
        mask_id = self.adapter.mask_token_id
        device = self.adapter.device
        gen_len = self.cfg.max_gen_len
        cfg_scale = self.cfg.cfg_weight

        prompt_ids_list = [self.adapter.encode(p) for p in prompts]
        prompt_len = max(ids.shape[1] for ids in prompt_ids_list)
        gen_mask = torch.full((len(prompts), gen_len), mask_id, dtype=torch.long, device=device)

        padded = torch.full((len(prompts), prompt_len), mask_id, dtype=torch.long, device=device)
        for i, ids in enumerate(prompt_ids_list):
            L = ids.shape[1]
            padded[i, :L] = ids[0]

        input_ids = torch.cat([padded, gen_mask], dim=1)   # [B, prompt_len + gen_len]

        # Build prompt_mask: True at prompt positions (so unconditional replaces with MASK)
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, ids in enumerate(prompt_ids_list):
            prompt_mask[i, : ids.shape[1]] = True

        def get_logits(x):
            return self._cfg_logits(x, prompt_mask, cfg_scale)

        with torch.no_grad():
            output_ids = run_denoising_loop(
                adapter=self.adapter,
                input_ids=input_ids,
                nfe=nfe,
                get_logits=get_logits,
                temperature=self.cfg.temperature,
                seed=self.cfg.seed,
            )

        generated = output_ids[:, prompt_len:]
        return [self.adapter.decode(generated[i]) for i in range(len(prompts))]

# STUB — Phase 2
# Implements: guided = full_logits + w * (full_logits - weak_logits)
# This is the core autoguidance step. Implement after Phase 1 is green.
#
# Design: GuidedSampler(adapter, weak_self, cfg) wraps the denoising loop and
# calls both adapter.logits(x) and weak_self(x, cond, adapter) at each step,
# combines them with the guidance formula, then passes guided logits to the
# unmasking step. Sweep w ∈ {0.5, 1.0, 1.5, 2.0, 3.0} and p ∈ {0.1, 0.2, 0.3}.
from autoguidance.samplers.base import SamplerBase


class GuidedSampler(SamplerBase):
    def __init__(self, adapter, cfg):
        raise NotImplementedError(
            "Phase 2: guided_step not implemented. "
            "Complete Phase 1 and get baselines before implementing guidance."
        )

    def sample(self, prompts, nfe):
        raise NotImplementedError

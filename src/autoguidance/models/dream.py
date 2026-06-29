# STUB — Phase 2
# Dream-org/Dream-v0-Instruct-7B adapter.
# Dream exposes generation_logits_hook_func which makes injecting guided logits
# easy without editing model internals. Implement in Phase 2 after Phase 1 is green.
from autoguidance.models.base import ModelAdapter


class DreamAdapter(ModelAdapter):
    def __init__(self, cfg):
        raise NotImplementedError(
            "Phase 2: Dream adapter not implemented. "
            "Use LLaDA for Phase 0 and Phase 1."
        )

    def logits(self, x_t, cond=None):
        raise NotImplementedError

    def embed(self, x_t):
        raise NotImplementedError

    def logits_from_embed(self, embeds, cond=None):
        raise NotImplementedError

    @property
    def mask_token_id(self):
        raise NotImplementedError

    @property
    def vocab_size(self):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

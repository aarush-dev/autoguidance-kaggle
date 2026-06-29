from autoguidance.models.base import ModelAdapter
from autoguidance.models.synthetic import SyntheticAdapter


def load_adapter(model_type: str, cfg) -> ModelAdapter:
    if model_type == "synthetic":
        return SyntheticAdapter(cfg)
    if model_type == "llada":
        from autoguidance.models.llada import LLaDAAdapter
        return LLaDAAdapter(cfg)
    if model_type == "dream":
        from autoguidance.models.dream import DreamAdapter
        return DreamAdapter(cfg)
    if model_type == "diffusiongemma":
        from autoguidance.models.diffusiongemma import DiffusionGemmaAdapter
        return DiffusionGemmaAdapter(cfg)
    raise ValueError(f"Unknown model_type: {model_type}")

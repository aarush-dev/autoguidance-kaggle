"""Weak-self factory.

Maps construction names → WeakSelf objects per the WEAK_SELF CONTRACT. Each
construction that needs an unsupported capability (e.g. reduced-expert on a
non-MoE model, layer-drop where layers can't be resolved, Gauss where embed
noise is unsupported) raises NotImplementedError at call time; run_phase0
catches and skips those.
"""
from autoguidance.weak_self.base import WeakSelf
from autoguidance.weak_self.input_noise import InputNoiseRemask, InputNoiseGauss
from autoguidance.weak_self.low_nfe import LowNFEWeakSelf
from autoguidance.weak_self.layer_drop import LayerDropWeakSelf
from autoguidance.weak_self.inference_dropout import InferenceDropoutWeakSelf
from autoguidance.weak_self.reduced_expert import ReducedExpertWeakSelf


def load_weak_self(name: str, cfg) -> WeakSelf:
    if name == "input_noise_remask":
        return InputNoiseRemask(remask_rate=cfg.noise_rate_remask, seed=cfg.seed)
    if name == "input_noise_gauss":
        return InputNoiseGauss(sigma=cfg.noise_sigma_gauss, seed=cfg.seed)
    if name == "low_nfe":
        return LowNFEWeakSelf(nfe=cfg.low_nfe_steps)
    if name == "layer_drop":
        return LayerDropWeakSelf(k=cfg.layer_drop_k)
    if name == "inference_dropout":
        return InferenceDropoutWeakSelf(p=cfg.dropout_p, seed=cfg.seed)
    if name == "reduced_expert":
        return ReducedExpertWeakSelf(top_k=cfg.reduced_expert_topk)
    raise ValueError(f"Unknown weak_self: {name}")

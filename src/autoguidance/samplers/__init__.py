from autoguidance.samplers.base import SamplerBase, SamplerConfig, masked_diffusion_step
from autoguidance.samplers.unguided import UnguidedSampler
from autoguidance.samplers.cfg_sampler import CFGSampler
from autoguidance.samplers.acfg_sampler import ACFGSampler


def load_sampler(name: str, adapter, cfg) -> SamplerBase:
    sampler_cfg = SamplerConfig(
        nfe=cfg.nfe_ladder[0] if hasattr(cfg, "nfe_ladder") else 128,
        cfg_weight=getattr(cfg, "cfg_weight", 2.0),
        acfg_threshold=getattr(cfg, "acfg_threshold", 0.3),
        temperature=0.0,
        seed=cfg.seed,
        max_gen_len=cfg.max_seq_len,
    )
    if name == "unguided":
        return UnguidedSampler(adapter, sampler_cfg)
    if name == "cfg":
        return CFGSampler(adapter, sampler_cfg)
    if name == "acfg":
        return ACFGSampler(adapter, sampler_cfg)
    if name == "guided":
        from autoguidance.samplers.guided_step import GuidedSampler
        return GuidedSampler(adapter, sampler_cfg)
    raise ValueError(f"Unknown sampler: {name}")

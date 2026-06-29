"""Phase 1 pipeline integration test on synthetic model (no GPU)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
import pytest

from autoguidance.models.synthetic import SyntheticAdapter, SEQ_LEN, MASK_ID, VOCAB_SIZE
from autoguidance.samplers.base import SamplerConfig, run_denoising_loop
from autoguidance.samplers.unguided import UnguidedSampler
from autoguidance.samplers.cfg_sampler import CFGSampler
from autoguidance.samplers.acfg_sampler import ACFGSampler
from autoguidance.phase1.report import save_results
from autoguidance.config import Phase1Config
import torch


@pytest.fixture
def adapter():
    return SyntheticAdapter(seed=42)


@pytest.fixture
def sampler_cfg():
    return SamplerConfig(
        nfe=4,
        cfg_weight=1.0,
        acfg_threshold=0.1,
        temperature=0.0,
        seed=42,
        max_gen_len=SEQ_LEN // 2,
    )


def test_denoising_loop_fills_masks(adapter):
    B, L = 2, SEQ_LEN
    x = torch.randint(0, VOCAB_SIZE - 1, (B, L // 2))
    gen_mask = torch.full((B, L // 2), MASK_ID, dtype=torch.long)
    input_ids = torch.cat([x, gen_mask], dim=1)

    out = run_denoising_loop(
        adapter=adapter,
        input_ids=input_ids,
        nfe=4,
        get_logits=lambda x_: adapter.logits(x_),
        temperature=0.0,
        seed=0,
    )
    assert out.shape == (B, L)
    # All MASK tokens should be filled
    assert (out == MASK_ID).sum() == 0


def test_unguided_sampler(adapter, sampler_cfg):
    sampler = UnguidedSampler(adapter, sampler_cfg)
    prompts = ["1 2 3 4 5", "10 20 30"]
    outputs = sampler.sample(prompts, nfe=4)
    assert len(outputs) == 2
    for out in outputs:
        assert isinstance(out, str)


def test_cfg_sampler(adapter, sampler_cfg):
    sampler = CFGSampler(adapter, sampler_cfg)
    prompts = ["1 2 3 4 5"]
    outputs = sampler.sample(prompts, nfe=4)
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)


def test_acfg_sampler(adapter, sampler_cfg):
    sampler = ACFGSampler(adapter, sampler_cfg)
    prompts = ["1 2 3 4 5"]
    outputs = sampler.sample(prompts, nfe=4)
    assert len(outputs) == 1


def test_phase1_report_smoke():
    rows = [
        {"decoder": "unguided", "nfe": 4, "gen_ppl": 100.0, "task_acc": 0.1,
         "distinct1": 0.5, "distinct2": 0.3, "self_bleu": 0.2, "mauve": 0.7,
         "elapsed_s": 1.0, "n_generated": 10, "seed": 42},
        {"decoder": "cfg", "nfe": 4, "gen_ppl": 90.0, "task_acc": 0.15,
         "distinct1": 0.4, "distinct2": 0.25, "self_bleu": 0.3, "mauve": 0.65,
         "elapsed_s": 2.0, "n_generated": 10, "seed": 42},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_results(rows, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "results.csv"))
        assert os.path.exists(os.path.join(tmpdir, "pareto.png"))


def test_samplers_deterministic(adapter, sampler_cfg):
    """Same seed → same output."""
    sampler = UnguidedSampler(adapter, sampler_cfg)
    prompts = ["5 10 15"]
    out1 = sampler.sample(prompts, nfe=4)
    out2 = sampler.sample(prompts, nfe=4)
    assert out1 == out2

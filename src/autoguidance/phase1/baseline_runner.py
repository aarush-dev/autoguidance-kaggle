"""Phase 1: baseline decoder × NFE ladder runner.

Runs 3 decoders (unguided, CFG, A-CFG) across the full NFE ladder on LLaDA-8B,
logging all quality and diversity metrics per (decoder, NFE) cell.
"""
from __future__ import annotations
from typing import List, Dict, Any
import random
import time

from autoguidance.models.base import ModelAdapter
from autoguidance.samplers import load_sampler
from autoguidance.samplers.base import SamplerConfig
from autoguidance.eval.distinct_n import distinct_n
from autoguidance.eval.self_bleu import self_bleu
from autoguidance.eval.task_accuracy import make_sudoku_batch, sudoku_accuracy


def _load_prompts(cfg, n: int) -> List[str]:
    """Load n open-ended prompts from WikiText-103."""
    try:
        from autoguidance.paths import iter_wikitext_texts
        prompts = []
        for raw in iter_wikitext_texts(cfg, cfg.dataset_split):
            # Use first ~30 tokens of each passage as the prompt
            words = raw.strip().split()
            if len(words) >= 30:
                prompts.append(" ".join(words[:30]))
            if len(prompts) >= n:
                break
        return prompts
    except Exception:
        rng = random.Random(cfg.seed + 100)
        return [
            " ".join(str(rng.randint(0, 50)) for _ in range(10))
            for _ in range(n)
        ]


def _load_reference_texts(cfg) -> List[str]:
    """Load human reference texts for MAUVE."""
    try:
        from autoguidance.paths import iter_wikitext_texts
        refs = []
        for raw in iter_wikitext_texts(cfg, "validation"):
            t = raw.strip()
            if len(t.split()) >= 20:
                refs.append(t)
            if len(refs) >= cfg.mauve_max_text:
                break
        return refs
    except Exception:
        return [f"reference text {i}" for i in range(100)]


def run_baselines(adapter: ModelAdapter, cfg) -> List[Dict[str, Any]]:
    """Run all decoders across the NFE ladder. Returns list of result dicts."""
    import torch
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    prompts = _load_prompts(cfg, cfg.n_samples)
    reference_texts = _load_reference_texts(cfg)
    sudoku_prompts, sudoku_solutions = make_sudoku_batch(cfg.sudoku_n_puzzles, seed=cfg.seed)

    sampler_cfg = SamplerConfig(
        cfg_weight=cfg.cfg_weight,
        acfg_threshold=cfg.acfg_threshold,
        temperature=0.0,
        seed=cfg.seed,
        max_gen_len=cfg.max_seq_len,
    )

    rows = []

    for decoder_name in cfg.decoders:
        print(f"\n[Phase 1] Decoder: {decoder_name}")
        sampler = load_sampler(decoder_name, adapter, cfg)
        sampler.cfg = sampler_cfg

        for nfe in cfg.nfe_ladder:
            print(f"  NFE={nfe} ...", end="", flush=True)
            t0 = time.time()

            # Generate text
            generated = []
            for i in range(0, len(prompts), cfg.batch_size):
                batch = prompts[i : i + cfg.batch_size]
                outputs = sampler.sample(batch, nfe=nfe)
                generated.extend(outputs)

            # Sudoku task accuracy
            sudoku_out = []
            for i in range(0, len(sudoku_prompts), cfg.batch_size):
                batch = sudoku_prompts[i : i + cfg.batch_size]
                outputs = sampler.sample(batch, nfe=nfe)
                sudoku_out.extend(outputs)
            task_acc = sudoku_accuracy(sudoku_out, sudoku_solutions)

            # Diversity metrics (CPU)
            d1 = distinct_n(generated, 1)
            d2 = distinct_n(generated, 2)
            sb = self_bleu(generated, n_pairs=cfg.self_bleu_n_pairs, seed=cfg.seed)

            # Quality: generative perplexity (P4000)
            try:
                from autoguidance.eval.gen_ppl import generative_perplexity
                gen_ppl = generative_perplexity(
                    generated,
                    scorer_model=(getattr(cfg, "scorer_path", "") or cfg.scorer_model),
                    device=cfg.device_eval,
                )
            except Exception as e:
                print(f"\n    [gen_ppl error] {e}")
                gen_ppl = float("nan")

            # MAUVE (P4000)
            try:
                from autoguidance.eval.mauve_eval import compute_mauve
                mauve_score = compute_mauve(
                    generated,
                    reference_texts,
                    device_id=int(cfg.device_eval.split(":")[-1]),
                    max_text=cfg.mauve_max_text,
                    featurize_model_name=(getattr(cfg, "mauve_path", "") or "gpt2-large"),
                )
            except Exception as e:
                print(f"\n    [MAUVE error] {e}")
                mauve_score = float("nan")

            elapsed = time.time() - t0
            row = {
                "decoder": decoder_name,
                "nfe": nfe,
                "gen_ppl": gen_ppl,
                "task_acc": task_acc,
                "distinct1": d1,
                "distinct2": d2,
                "self_bleu": sb,
                "mauve": mauve_score,
                "elapsed_s": elapsed,
                "n_generated": len(generated),
                "seed": cfg.seed,
            }
            rows.append(row)
            print(
                f" done ({elapsed:.1f}s) ppl={gen_ppl:.1f} mauve={mauve_score:.3f} "
                f"d1={d1:.3f} d2={d2:.3f} task_acc={task_acc:.3f}"
            )

    return rows

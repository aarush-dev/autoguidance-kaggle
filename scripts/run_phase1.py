#!/usr/bin/env python3
"""Phase 1 runner: baseline decoders x NFE ladder.

Usage:
    # CPU smoke test (synthetic model, fast)
    python scripts/run_phase1.py --model synthetic --n-samples 5 --nfe 4,8

    # Real run on LLaDA-8B (GPU required, ~6-12 hours for full ladder)
    python scripts/run_phase1.py --model llada

    # DiffusionGemma (confirm load API at build time before using)
    python scripts/run_phase1.py --model diffusiongemma

    # With custom config
    python scripts/run_phase1.py --config configs/phase1.yaml --model llada
"""
import sys
import os

import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoguidance.config import Phase1Config, load_config
from autoguidance.models import load_adapter
from autoguidance.phase1.baseline_runner import run_baselines
from autoguidance.phase1.report import save_results


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: baseline evaluation")
    p.add_argument("--config", default="configs/phase1.yaml")
    p.add_argument("--model", default=None, choices=["synthetic", "llada", "diffusiongemma"])
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--nfe", default=None, help="Comma-separated NFE values to run, e.g. '4,8,16'")
    p.add_argument("--output-dir", default="outputs/phase1")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.config):
        cfg = load_config(args.config, Phase1Config)
    else:
        print(f"[warn] Config not found at {args.config}, using defaults")
        cfg = Phase1Config()

    if args.n_samples:
        cfg.n_samples = args.n_samples
    if args.nfe:
        cfg.nfe_ladder = [int(x) for x in args.nfe.split(",")]

    model_type = args.model or "llada"

    print(f"[Phase 1] model={model_type} precision={cfg.precision} dtype={cfg.dtype}")
    print(f"[Phase 1] device_main={cfg.device_main} device_eval={cfg.device_eval}")
    print(f"[Phase 1] decoders={cfg.decoders}")
    print(f"[Phase 1] nfe_ladder={cfg.nfe_ladder}")
    print(f"[Phase 1] n_samples={cfg.n_samples}")

    adapter = load_adapter(model_type, cfg)
    rows = run_baselines(adapter, cfg)
    save_results(rows, args.output_dir)

    print(f"\n[Phase 1] DONE. Results in {args.output_dir}/")
    sys.exit(0)


if __name__ == "__main__":
    main()

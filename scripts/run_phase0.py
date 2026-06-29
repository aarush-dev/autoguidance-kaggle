#!/usr/bin/env python3
"""Phase 0 runner: weak-self characterization (bf16, single GPU cuda:0).

Flow:
    load adapter -> run_characterization (dump arrays) -> evaluate_dir (thresholds)
    -> [optional] construction-param sweeps -> save report -> exit code by any_pass.

Usage:
    # CPU smoke test (synthetic model, fast)
    python scripts/run_phase0.py --model synthetic --n-samples 10

    # Real run (bf16 on cuda:0), then sweep construction params
    python scripts/run_phase0.py --model llada --sweep

    # Re-threshold only — no model, reuse dumped arrays
    python scripts/run_phase0.py --thresholds-only --arrays-dir /kaggle/working/phase0_arrays

    # Custom config
    python scripts/run_phase0.py --config configs/phase0.yaml --model diffusiongemma
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoguidance.config import Phase0Config, load_config
from autoguidance.phase0.characterize import run_characterization
from autoguidance.phase0.thresholds import evaluate_dir, evaluate_arrays
from autoguidance.phase0.report import save_results


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0: weak-self characterization")
    p.add_argument("--config", default="configs/phase0.yaml", help="Path to phase0 config YAML")
    p.add_argument("--model", default=None,
                   choices=["synthetic", "llada", "dream", "diffusiongemma"],
                   help="Model type (default: inferred from transformers_variant)")
    p.add_argument("--precision", default=None, help="Override precision (default: bf16 from config)")
    p.add_argument("--n-samples", type=int, default=None, help="Override n_samples")
    p.add_argument("--output-dir", default="outputs/phase0", help="Report output directory")
    p.add_argument("--arrays-dir", default=None, help="Where reduced arrays live (default: cfg.arrays_dir)")
    p.add_argument("--sweep", action="store_true", help="Run construction-parameter sweeps")
    p.add_argument("--thresholds-only", action="store_true",
                   help="Skip the model; just re-evaluate saved arrays")
    return p.parse_args()


def _infer_model(cfg) -> str:
    variant = getattr(cfg, "transformers_variant", "tf5")
    if variant == "tf446":
        return "llada"
    if variant == "tf5":
        return "diffusiongemma"
    return "synthetic"


def _run_sweeps(adapter, cfg, arrays_dir):
    """Run construction-parameter sweeps; return {sweep_name: rows} for the report."""
    from autoguidance.phase0.sweep import sweep_construction_param, PARAM_TO_CONSTRUCTION
    from autoguidance.phase0.arrays import load_arrays

    sweep_dir = os.path.join(arrays_dir, "sweeps")
    grid = {
        "noise_rate_remask": getattr(cfg, "sweep_remask_rate", []),
        "noise_sigma_gauss": getattr(cfg, "sweep_gauss_sigma", []),
        "layer_drop_k": getattr(cfg, "sweep_layer_drop_k", []),
        "dropout_p": getattr(cfg, "sweep_dropout_p", []),
    }
    full_cache = {}   # shared across sweeps so the full pass runs once
    sweep_results = {}
    for param_name, values in grid.items():
        if not values or param_name not in PARAM_TO_CONSTRUCTION:
            continue
        produced = sweep_construction_param(
            adapter, cfg, param_name, list(values), sweep_dir, full_cache=full_cache
        )
        rows = []
        for value, path in produced.items():
            stem = os.path.splitext(os.path.basename(path))[0]
            v = evaluate_arrays(load_arrays(sweep_dir, stem), cfg)
            rows.append({
                "param": param_name,
                "value": value,
                "check1_pass": bool(v["check1"]["check1_pass"]),
                "check2_pass": bool(v["check2"]["check2_pass"]),
                "check3_pass": bool(v["check3"]["check3_pass"]),
                "all_pass": bool(v["all_pass"]),
                "top1_agreement": float(v["check2"]["top1_agreement"]),
                "spearman_rho": float(v["check2"]["spearman_rho_mean"]),
                "pearson_r": float(v["check3"]["pearson_r"]),
                "precision": float(v["check3"]["precision"]),
            })
        if rows:
            sweep_results[param_name] = rows
    return sweep_results


def main():
    args = parse_args()

    if os.path.exists(args.config):
        cfg = load_config(args.config, Phase0Config)
    else:
        print(f"[warn] Config not found at {args.config}, using defaults")
        cfg = Phase0Config()

    if args.precision:
        cfg.precision = args.precision
    if args.n_samples:
        cfg.n_samples = args.n_samples

    arrays_dir = args.arrays_dir or getattr(cfg, "arrays_dir", "phase0_arrays")

    print("=" * 70)
    print("[Phase 0] runner")
    print(f"  device_main={getattr(cfg, 'device_main', 'cuda:0')} "
          f"precision={getattr(cfg, 'precision', 'bf16')} "
          f"dtype={getattr(cfg, 'dtype', 'bfloat16')}")
    print(f"  n_samples={cfg.n_samples} mask_rate={cfg.mask_rate}")
    print(f"  constructions={cfg.constructions}")
    print(f"  arrays_dir={arrays_dir} output_dir={args.output_dir}")
    print(f"  thresholds_only={args.thresholds_only} sweep={args.sweep}")
    print("=" * 70)

    # --- thresholds-only: no model, just re-evaluate dumped arrays ---
    if args.thresholds_only:
        verdicts = evaluate_dir(arrays_dir, cfg)
        save_results(verdicts, args.output_dir)
        _finish(verdicts)
        return

    # --- full pipeline ---
    from autoguidance.models import load_adapter
    from autoguidance.weak_self import load_weak_self

    model_type = args.model or _infer_model(cfg)
    print(f"[Phase 0] loading adapter: model={model_type}")
    adapter = load_adapter(model_type, cfg)

    weak_selfs = {}
    for name in cfg.constructions:
        try:
            weak_selfs[name] = load_weak_self(name, cfg)
        except NotImplementedError as e:
            print(f"[Phase 0] Skipping {name}: {e}")
    if not weak_selfs:
        print("[Phase 0] ERROR: no valid constructions to run")
        sys.exit(1)

    run_characterization(adapter, weak_selfs, cfg, arrays_dir)
    verdicts = evaluate_dir(arrays_dir, cfg)

    sweep_results = None
    if args.sweep:
        sweep_results = _run_sweeps(adapter, cfg, arrays_dir)

    save_results(verdicts, args.output_dir, sweep_results=sweep_results)
    _finish(verdicts)


def _finish(verdicts):
    any_pass = any(v.get("all_pass", False) for v in verdicts.values())
    if not any_pass:
        print("\n[Phase 0] FAILED: no construction passed all 3 checks. "
              "Report this as a negative result.")
        sys.exit(1)
    print("\n[Phase 0] DONE. At least one construction passed all checks. Proceed to Phase 1.")
    sys.exit(0)


if __name__ == "__main__":
    main()

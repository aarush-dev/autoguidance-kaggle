#!/usr/bin/env python3
"""Phase 0 threshold evaluator — NO model.

Reloads the reduced arrays dumped by characterize.py and re-applies the
check1/check2/check3 criteria, optionally sweeping thresholds. Cheap, pure numpy:
change a threshold and re-run this instead of the expensive characterization.

Usage:
    python scripts/eval_thresholds.py --arrays-dir /kaggle/working/phase0_arrays
    python scripts/eval_thresholds.py --arrays-dir DIR --config configs/phase0.yaml --sweep
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoguidance.config import Phase0Config, load_config
from autoguidance.phase0.thresholds import evaluate_dir
from autoguidance.phase0.sweep import sweep_threshold
from autoguidance.phase0.report import save_results


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0: evaluate thresholds on saved arrays (no model)")
    p.add_argument("--arrays-dir", required=True, help="Directory of dumped construction npz files")
    p.add_argument("--config", default="configs/phase0.yaml", help="Path to phase0 config YAML")
    p.add_argument("--output-dir", default="outputs/phase0", help="Report output directory")
    p.add_argument("--sweep", action="store_true", help="Also sweep check2/check3 thresholds")
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.config):
        cfg = load_config(args.config, Phase0Config)
    else:
        print(f"[warn] Config not found at {args.config}, using defaults")
        cfg = Phase0Config()

    print("=" * 70)
    print("[Phase 0] eval_thresholds (no model)")
    print(f"  arrays_dir={args.arrays_dir} output_dir={args.output_dir} sweep={args.sweep}")
    print("=" * 70)

    verdicts = evaluate_dir(args.arrays_dir, cfg)

    sweep_results = None
    if args.sweep:
        grid = {
            "check2_top1_agreement": [0.50, 0.60, 0.70],
            "check2_spearman_rho": [0.40, 0.50, 0.60],
            "check3_pearson_r": [0.10, 0.20, 0.30],
            "check3_precision": [0.40, 0.50, 0.60],
        }
        rows = sweep_threshold(args.arrays_dir, grid, cfg)
        sweep_results = {"thresholds": rows}

    save_results(verdicts, args.output_dir, sweep_results=sweep_results)

    any_pass = any(v.get("all_pass", False) for v in verdicts.values())
    if not any_pass:
        print("\n[Phase 0] FAILED: no construction passed all 3 checks.")
        sys.exit(1)
    print("\n[Phase 0] DONE. At least one construction passed all checks.")
    sys.exit(0)


if __name__ == "__main__":
    main()

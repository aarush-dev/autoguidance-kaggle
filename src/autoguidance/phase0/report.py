"""Phase 0: generate and save report artifacts."""
from __future__ import annotations
from typing import Dict, Any
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    sweep_results: Dict[str, Any] = None,
) -> None:
    """Save all Phase 0 output artifacts.

    Artifacts:
        results.json           — full structured results (verdicts)
        agreement_table.md     — pass/fail table
        entropy_dist.png       — histogram of per-position entropy ratios
        error_position_scatter.png — scatter of disagreement vs full-model error
        verdict.txt            — which constructions pass all 3 checks
        sweep_<name>.md        — one markdown table per provided sweep (optional)

    Args:
        results: {construction: verdict dict} from thresholds.evaluate_dir /
            evaluate_arrays (each carries check1/check2/check3/all_pass).
        output_dir: where artifacts are written.
        sweep_results: optional {sweep_name: list[row dict]} — each rendered as
            sweep_<sweep_name>.md.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. results.json (serialize numpy arrays to lists)
    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_to_serializable(results), f, indent=2)
    print(f"[Phase 0] Saved {results_path}")

    # 2. agreement_table.md
    table_lines = [
        "# Phase 0: Weak-Self Characterization — Pass/Fail Table\n",
        "| Construction | Check 1 (Entropy) | Check 2 (Agreement) | Check 3 (Error pos.) | All Pass |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        if "error" in r:
            table_lines.append(f"| {name} | ERROR | ERROR | ERROR | FAIL |")
            continue
        c1 = "PASS" if r["check1"]["check1_pass"] else "FAIL"
        c2 = "PASS" if r["check2"]["check2_pass"] else "FAIL"
        c3 = "PASS" if r["check3"]["check3_pass"] else "FAIL"
        ap = "**PASS**" if r["all_pass"] else "FAIL"
        table_lines.append(f"| {name} | {c1} | {c2} | {c3} | {ap} |")

    table_path = os.path.join(output_dir, "agreement_table.md")
    with open(table_path, "w") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"[Phase 0] Saved {table_path}")

    # 3. entropy_dist.png
    fig, axes = plt.subplots(1, max(1, len(results)), figsize=(5 * max(1, len(results)), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        if "error" in r or "check1" not in r:
            ax.set_title(f"{name}\n(ERROR)")
            continue
        delta = r["check1"]["delta_entropy"]
        ax.hist(delta, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Δ=0")
        ax.set_title(f"{name}\nΔH = H_weak − H_full")
        ax.set_xlabel("ΔH (nats)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    entropy_path = os.path.join(output_dir, "entropy_dist.png")
    plt.savefig(entropy_path, dpi=120)
    plt.close()
    print(f"[Phase 0] Saved {entropy_path}")

    # 4. error_position_scatter.png
    fig, axes = plt.subplots(1, max(1, len(results)), figsize=(5 * max(1, len(results)), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        if "error" in r or "check3" not in r:
            ax.set_title(f"{name}\n(ERROR)")
            continue
        c3 = r["check3"]
        ax.set_title(
            f"{name}\nr={c3['pearson_r']:.3f} prec={c3['precision']:.3f} "
            f"{'PASS' if c3['check3_pass'] else 'FAIL'}"
        )
        ax.set_xlabel("Full-model error rate")
        ax.set_ylabel("Disagreement rate")
        ax.text(0.5, 0.5, f"r={c3['pearson_r']:.3f}", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "error_position_scatter.png")
    plt.savefig(scatter_path, dpi=120)
    plt.close()
    print(f"[Phase 0] Saved {scatter_path}")

    # 5. verdict.txt
    passing = [name for name, r in results.items() if r.get("all_pass", False)]
    verdict_lines = ["# Phase 0 Verdict\n"]
    if passing:
        verdict_lines.append(f"Constructions passing all 3 checks: {', '.join(passing)}")
        verdict_lines.append("\nRecommendation: Proceed to Phase 1 using the passing constructions.")
        verdict_lines.append("Recommended for Phase 2 guidance: " + passing[0])
    else:
        verdict_lines.append("NO construction passed all 3 checks.")
        verdict_lines.append("\nInterpretation: The weak-self constructions tested do not satisfy")
        verdict_lines.append("the 'same errors, amplified' precondition for autoguidance.")
        verdict_lines.append("STOP — report this as a negative result before proceeding.")

    verdict_lines.append("\n## Per-construction details")
    for name, r in results.items():
        if "error" in r:
            verdict_lines.append(f"\n### {name}: ERROR\n  {r['error']}")
            continue
        verdict_lines.append(f"\n### {name}: {'ALL PASS' if r['all_pass'] else 'FAIL'}")
        c1 = r["check1"]
        degen = " [DEGENERATE full entropy — position misalignment suspected, result invalid]" \
            if c1.get("degenerate_full_entropy") else ""
        verdict_lines.append(
            f"  Check 1 (Entropy): {'PASS' if c1['check1_pass'] else 'FAIL'} "
            f"— mean_full={c1['mean_full']:.3f} mean_weak={c1['mean_weak']:.3f} "
            f"fraction_higher={c1['fraction_weak_higher']:.3f}{degen}"
        )
        c2 = r["check2"]
        verdict_lines.append(
            f"  Check 2 (Agreement): {'PASS' if c2['check2_pass'] else 'FAIL'} "
            f"— top1_agreement={c2['top1_agreement']:.3f} "
            f"spearman_rho={c2['spearman_rho_mean']:.3f}"
        )
        c3 = r["check3"]
        verdict_lines.append(
            f"  Check 3 (Error pos.): {'PASS' if c3['check3_pass'] else 'FAIL'} "
            f"— pearson_r={c3['pearson_r']:.3f} precision={c3['precision']:.3f}"
        )

    verdict_path = os.path.join(output_dir, "verdict.txt")
    with open(verdict_path, "w") as f:
        f.write("\n".join(verdict_lines) + "\n")
    print(f"[Phase 0] Saved {verdict_path}")
    print("\n" + "\n".join(verdict_lines))

    # 6. sweep_<name>.md (optional)
    if sweep_results:
        for sweep_name, rows in sweep_results.items():
            _write_sweep_table(sweep_name, rows, output_dir)


def _write_sweep_table(sweep_name: str, rows, output_dir: str) -> None:
    """Render a list of row dicts as a markdown table sweep_<sweep_name>.md."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(sweep_name))
    path = os.path.join(output_dir, f"sweep_{safe}.md")
    lines = [f"# Phase 0 Sweep: {sweep_name}\n"]
    if not rows:
        lines.append("_(no rows)_")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[Phase 0] Saved {path}")
        return

    cols = list(rows[0].keys())

    def _cell(v):
        if isinstance(v, bool):
            return "PASS" if v else "FAIL"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_cell(r.get(c, "")) for c in cols) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Phase 0] Saved {path}")

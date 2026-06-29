"""Phase 1: save CSV and Pareto plot."""
from __future__ import annotations
from typing import List, Dict, Any
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


DECODERS = ["unguided", "cfg", "acfg"]
COLORS = {"unguided": "#4477AA", "cfg": "#EE6677", "acfg": "#228833"}
MARKERS = {"unguided": "o", "cfg": "s", "acfg": "^"}


def save_results(rows: List[Dict[str, Any]], output_dir: str) -> None:
    """Save results.csv and pareto.png."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. results.csv
    csv_path = os.path.join(output_dir, "results.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"[Phase 1] Saved {csv_path}")

    # 2. pareto.png — quality (gen_ppl) vs diversity (MAUVE)
    fig, ax = plt.subplots(figsize=(8, 6))
    nfe_values = sorted(set(r["nfe"] for r in rows))

    # Marker size proportional to NFE (log scale)
    nfe_arr = np.array(nfe_values, dtype=float)
    size_map = {n: 40 + 120 * (np.log(n) / np.log(max(nfe_arr))) for n in nfe_values}

    for decoder in DECODERS:
        d_rows = [r for r in rows if r["decoder"] == decoder]
        if not d_rows:
            continue
        x = [r["mauve"] for r in d_rows]
        y = [r["gen_ppl"] for r in d_rows]
        sizes = [size_map.get(r["nfe"], 80) for r in d_rows]

        ax.scatter(
            x, y,
            s=sizes,
            color=COLORS.get(decoder, "gray"),
            marker=MARKERS.get(decoder, "o"),
            label=decoder,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.5,
        )
        # Connect NFE points with a line
        sorted_rows = sorted(d_rows, key=lambda r: r["nfe"])
        ax.plot(
            [r["mauve"] for r in sorted_rows],
            [r["gen_ppl"] for r in sorted_rows],
            color=COLORS.get(decoder, "gray"),
            linewidth=1.2,
            alpha=0.5,
        )

    ax.set_xlabel("MAUVE (diversity ↑)", fontsize=12)
    ax.set_ylabel("Generative Perplexity (quality ↓)", fontsize=12)
    ax.set_title("Phase 1 Baselines: Quality–Diversity Pareto Frontier\n(marker size ∝ NFE)", fontsize=11)
    ax.legend(title="Decoder", fontsize=10)
    ax.invert_yaxis()   # lower perplexity = better quality = higher on plot
    ax.grid(alpha=0.3)

    # NFE legend
    legend_handles = [
        plt.scatter([], [], s=size_map[n], color="gray", label=f"NFE={n}")
        for n in nfe_values
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_handles,
              labels=ax.get_legend_handles_labels()[1] + [f"NFE={n}" for n in nfe_values],
              fontsize=8, ncol=2)

    plt.tight_layout()
    pareto_path = os.path.join(output_dir, "pareto.png")
    plt.savefig(pareto_path, dpi=150)
    plt.close()
    print(f"[Phase 1] Saved {pareto_path}")

    # Print summary table
    print("\n[Phase 1] Results summary:")
    print(f"{'Decoder':<12} {'NFE':>6} {'PPL':>8} {'MAUVE':>7} {'D1':>6} {'D2':>6} {'SelfBLEU':>9} {'TaskAcc':>9}")
    for r in rows:
        print(
            f"{r['decoder']:<12} {r['nfe']:>6} {r['gen_ppl']:>8.2f} "
            f"{r['mauve']:>7.4f} {r['distinct1']:>6.3f} {r['distinct2']:>6.3f} "
            f"{r['self_bleu']:>9.4f} {r['task_acc']:>9.4f}"
        )

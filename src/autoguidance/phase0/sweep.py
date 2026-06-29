"""Phase 0: sweeps.

Two flavors, both built to avoid re-running the expensive full-model pass:

1. sweep_threshold(arrays_dir, grid, cfg)
   Post-step only. Re-reads the cached arrays and re-applies check1/check2/check3
   under a grid of *threshold* values. Pure numpy, no model.

2. sweep_construction_param(adapter, cfg, param_name, values, arrays_dir, full_cache)
   Re-runs ONLY the weak-self pass for each construction-parameter value. The
   full-model pass (entropy_full, top1_full, gt + full masked logits needed for
   diff_mag / rho) is computed once and cached keyed by (model, seed, mask_rate);
   subsequent values reuse it. Dumps one npz per value via save_arrays.
"""
from __future__ import annotations
from typing import Dict, List
import dataclasses
import itertools

import numpy as np
import torch

from autoguidance.phase0.arrays import save_arrays, load_arrays, list_constructions
from autoguidance.phase0.thresholds import evaluate_arrays
from autoguidance.phase0.characterize import (
    _load_dataset_texts,
    _mask_positions,
    _reduce_position_arrays,
    _collect_rhos,
    _EMBED_NOISE_CONSTRUCTIONS,
)

# Sweep parameter (a Phase0Config field) -> construction it parameterizes.
PARAM_TO_CONSTRUCTION = {
    "noise_rate_remask": "input_noise_remask",
    "noise_sigma_gauss": "input_noise_gauss",
    "layer_drop_k": "layer_drop",
    "dropout_p": "inference_dropout",
    "low_nfe_steps": "low_nfe",
    "reduced_expert_topk": "reduced_expert",
}


def _replace_cfg(cfg, **overrides):
    """Override fields on a (dataclass) cfg, falling back to mutation for non-dataclasses."""
    if dataclasses.is_dataclass(cfg):
        return dataclasses.replace(cfg, **overrides)
    import copy
    c = copy.copy(cfg)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def sweep_threshold(arrays_dir: str, grid: Dict[str, List], cfg) -> List[Dict]:
    """Re-evaluate cached arrays across a grid of threshold values (no model).

    Args:
        arrays_dir: directory of cached construction npz files.
        grid: {threshold_field: [values]} — Cartesian product is taken.
        cfg: base Phase0Config (threshold fields are overridden per combo).

    Returns:
        list of row dicts (one per construction x threshold-combo).
    """
    names = list_constructions(arrays_dir)
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys])) if keys else [()]

    print(f"\n[Phase 0] sweep_threshold: {arrays_dir}")
    print(f"  constructions={names}")
    print(f"  grid={grid}  ({len(combos)} combos x {len(names)} constructions)")

    rows: List[Dict] = []
    for name in names:
        arrays = load_arrays(arrays_dir, name)
        for combo in combos:
            overrides = dict(zip(keys, combo))
            cfg2 = _replace_cfg(cfg, **overrides)
            v = evaluate_arrays(arrays, cfg2)
            row = {"construction": name}
            row.update(overrides)
            row.update({
                "check1_pass": bool(v["check1"]["check1_pass"]),
                "check2_pass": bool(v["check2"]["check2_pass"]),
                "check3_pass": bool(v["check3"]["check3_pass"]),
                "all_pass": bool(v["all_pass"]),
                "top1_agreement": float(v["check2"]["top1_agreement"]),
                "spearman_rho": float(v["check2"]["spearman_rho_mean"]),
                "pearson_r": float(v["check3"]["pearson_r"]),
                "precision": float(v["check3"]["precision"]),
            })
            rows.append(row)

    _print_rows(rows, title="sweep_threshold results")
    return rows


def _build_full_cache(adapter, cfg, texts, spearman_cap, per_sample_rho):
    """Run the full-model pass once; cache per-sample data the weak pass will reuse.

    Caches per surviving sample:
        ids   — masked input ids (CPU long, [1, L])
        m     — mask indicator (CPU bool, [L])
        f_mask— full-model logits at masked positions (CPU fp16, [n_masked, vocab])
        gt    — ground-truth ids at masked positions (numpy)
    Storing the full masked logits (fp16) is what lets diff_mag / rho be recomputed
    against any future weak pass WITHOUT re-running the full model.
    """
    samples = []
    n_bytes = 0
    import random
    rng = random.Random(cfg.seed)
    for text in texts:
        try:
            token_ids = adapter.encode(text)
            if token_ids.shape[1] < 4:
                continue
            gt_ids, mask_ind = _mask_positions(token_ids, cfg.mask_rate, rng)
            masked_ids = adapter.corrupt_positions(
                gt_ids.to(adapter.device), mask_ind.to(adapter.device)
            )
            m = mask_ind.squeeze(0)
            with torch.no_grad():
                f_logits = adapter.logits(masked_ids)[0]
                mdev = m.to(f_logits.device)
                f_mask = f_logits[mdev].float().cpu().half()   # [n_masked, vocab] fp16 on CPU
            gt_masked = gt_ids.squeeze(0).cpu().numpy()[m.numpy()]
            samples.append({
                "ids": masked_ids.cpu(),
                "m": m,
                "f_mask": f_mask,
                "gt": gt_masked,
            })
            n_bytes += f_mask.numel() * 2
            del f_logits
        except Exception as e:
            print(f"    [full_cache] skipping sample: {e}")
            continue
    print(f"  [full_cache] cached {len(samples)} samples, "
          f"full masked logits ~{n_bytes / 1e9:.2f} GB (fp16 CPU)")
    return {
        "samples": samples,
        "spearman_cap": spearman_cap,
        "per_sample_rho": per_sample_rho,
    }


def sweep_construction_param(
    adapter,
    cfg,
    param_name: str,
    values: List,
    arrays_dir: str,
    full_cache: Dict = None,
) -> Dict:
    """Sweep one construction parameter, re-running only the weak-self pass per value.

    Args:
        adapter: full model adapter.
        cfg: base Phase0Config (param_name is overridden per value).
        param_name: a Phase0Config field in PARAM_TO_CONSTRUCTION.
        values: list of values to try.
        arrays_dir: where to dump one npz per value.
        full_cache: optional dict reused across calls; keyed by (model, seed, mask_rate).
            Pass the same dict in across multiple sweeps to share the full pass.

    Returns:
        dict {value: npz_path}.
    """
    from autoguidance.weak_self import load_weak_self

    if param_name not in PARAM_TO_CONSTRUCTION:
        raise ValueError(
            f"Unknown sweep param '{param_name}'. Known: {sorted(PARAM_TO_CONSTRUCTION)}"
        )
    construction = PARAM_TO_CONSTRUCTION[param_name]

    print(f"\n[Phase 0] sweep_construction_param: {param_name} -> {construction}")
    print(f"  values={values}  arrays_dir={arrays_dir}")

    supports_embed = getattr(adapter, "supports_embed_noise", True)
    if construction in _EMBED_NOISE_CONSTRUCTIONS and not supports_embed:
        print(f"  [skip] {construction} needs embedding-noise support "
              f"(adapter.supports_embed_noise=False)")
        return {}

    spearman_cap = 2000
    per_sample_rho = max(1, spearman_cap // max(1, cfg.n_samples))

    if full_cache is None:
        full_cache = {}
    key = (type(adapter).__name__, cfg.seed, cfg.mask_rate)
    if key not in full_cache:
        print(f"  [full_cache] building for key={key} (one full-model pass)")
        texts = _load_dataset_texts(cfg, cfg.n_samples)
        full_cache[key] = _build_full_cache(
            adapter, cfg, texts, spearman_cap, per_sample_rho
        )
    else:
        print(f"  [full_cache] reusing cached full pass for key={key}")
    cache = full_cache[key]
    samples = cache["samples"]

    out: Dict = {}
    for v in values:
        cfg2 = _replace_cfg(cfg, **{param_name: v})
        try:
            weak_self = load_weak_self(construction, cfg2)
        except NotImplementedError as e:
            print(f"  [{param_name}={v}] skip: {e}")
            continue

        ent_full, ent_weak = [], []
        full_top1, weak_top1 = [], []
        diff_mag, gt_flat = [], []
        rhos: list = []
        n_processed = 0

        for entry in samples:
            try:
                masked_ids = entry["ids"].to(adapter.device)
                m = entry["m"]
                with torch.no_grad():
                    w_logits = weak_self(masked_ids, None, adapter)[0]
                    mdev = m.to(w_logits.device)
                    w_mask = w_logits[mdev].float()
                f_mask = entry["f_mask"].to(w_mask.device).float()

                ef, ew, ft, wt, dm = _reduce_position_arrays(f_mask, w_mask)
                ent_full.append(ef)
                ent_weak.append(ew)
                full_top1.append(ft)
                weak_top1.append(wt)
                diff_mag.append(dm)
                gt_flat.append(entry["gt"])
                _collect_rhos(f_mask, w_mask, per_sample_rho, rhos, spearman_cap)
                del w_logits, w_mask, f_mask
                n_processed += 1
            except Exception as e:
                print(f"    [{param_name}={v}] skipping sample: {e}")
                continue

        if n_processed == 0:
            print(f"  [{param_name}={v}] ERROR: no samples processed — skipping")
            continue

        ef = np.concatenate(ent_full)
        ew = np.concatenate(ent_weak)
        ft = np.concatenate(full_top1)
        wt = np.concatenate(weak_top1)
        dm = np.concatenate(diff_mag)
        gt = np.concatenate(gt_flat)
        rho_sample = np.asarray(rhos, dtype=np.float32)

        name = f"{construction}__{param_name}={v}"
        meta = {
            "construction": construction,
            "sweep_param": param_name,
            "sweep_value": v,
            "n_samples_processed": n_processed,
            "n_masked_positions": int(ef.shape[0]),
            "mask_rate": cfg.mask_rate,
            "seed": cfg.seed,
            "model": type(adapter).__name__,
            "n_rho_samples": int(rho_sample.shape[0]),
        }
        path = save_arrays(
            arrays_dir, name, meta,
            entropy_full=ef, entropy_weak=ew,
            top1_full=ft, top1_weak=wt,
            diff_mag=dm, gt=gt, rho_sample=rho_sample,
        )
        out[v] = path
        print(f"  [{param_name}={v}] dumped {ef.shape[0]} masked positions")

    return out


def _print_rows(rows: List[Dict], title: str = "results") -> None:
    if not rows:
        print(f"  ({title}: no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(_fmt(r.get(c, ""))) for r in rows)) for c in cols}
    print(f"\n  {title}:")
    print("  " + " | ".join(str(c).ljust(widths[c]) for c in cols))
    print("  " + "-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + " | ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in cols))


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)

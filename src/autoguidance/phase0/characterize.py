"""Phase 0: weak-self characterization (array-dumping pass).

For each construction this runs TWO forward passes per masked sample (full model +
weak self), reduces each masked position to a handful of scalars/ids on-GPU, and
DUMPS those reduced arrays to disk via arrays.save_arrays. It does NOT threshold —
pass/fail lives in thresholds.py so the expensive passes run once and the cheap
criteria can be re-evaluated / swept without a model.

Reduced per-masked-position arrays (the only things that ever cross to CPU/disk):
    entropy_full, entropy_weak, top1_full, top1_weak, diff_mag, gt, rho_sample

Memory rule (CLAUDE.md): never accumulate the [N, L, vocab] logit tensors
(200 x 256 x ~126k x bf16 > 10 GB per tensor). Reduce on-GPU, stream the scalars.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import random

import numpy as np
import torch
from torch import LongTensor

from autoguidance.models.base import ModelAdapter
from autoguidance.weak_self.base import WeakSelf
from autoguidance.eval.entropy import entropy_nats
from autoguidance.eval.agreement import spearman_rho

# Constructions that perturb the input embeddings; only runnable when the adapter
# exposes logits_from_embed (adapter.supports_embed_noise).
_EMBED_NOISE_CONSTRUCTIONS = {"input_noise_gauss"}


def _mask_positions(
    token_ids: LongTensor,     # [1, seq] or [seq]
    mask_rate: float,
    rng: random.Random,
) -> Tuple[LongTensor, "torch.BoolTensor"]:
    """Pick a random fraction of positions to corrupt.

    Returns (gt_ids[1, seq] long, mask_indicator[1, seq] bool). The actual token
    replacement is delegated to adapter.corrupt_positions so each model uses its
    own noise process.
    """
    ids = token_ids.squeeze(0)           # [seq]
    n = len(ids)
    positions = list(range(n))
    rng.shuffle(positions)
    n_to_mask = max(1, int(n * mask_rate))
    mask_pos = sorted(positions[:n_to_mask])

    mask_indicator = torch.zeros(n, dtype=torch.bool)
    mask_indicator[mask_pos] = True
    return ids.unsqueeze(0), mask_indicator.unsqueeze(0)


def _load_dataset_texts(cfg, n_samples: int) -> List[str]:
    """Load n_samples text strings from the configured dataset (wikitext)."""
    split = getattr(cfg, "dataset_split", "validation")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-v1", split=split, streaming=True)
        texts = []
        for ex in ds:
            t = ex["text"].strip()
            if len(t.split()) >= 20:   # skip very short lines
                texts.append(t)
            if len(texts) >= n_samples:
                break
        if texts:
            return texts
    except Exception as e:
        print(f"[Phase 0] dataset load failed ({e}); using synthetic fallback texts")
    rng = random.Random(cfg.seed)
    return [
        " ".join(str(rng.randint(0, 50)) for _ in range(20))
        for _ in range(n_samples)
    ]


def _reduce_position_arrays(f_mask, w_mask):
    """Reduce [n_masked, vocab] full/weak logits to per-position scalars/ids (numpy)."""
    ent_f = entropy_nats(f_mask).cpu().numpy()
    ent_w = entropy_nats(w_mask).cpu().numpy()
    top1_f = f_mask.argmax(dim=-1).cpu().numpy()
    top1_w = w_mask.argmax(dim=-1).cpu().numpy()
    diff = (f_mask - w_mask).abs().max(dim=-1).values.cpu().numpy()
    return ent_f, ent_w, top1_f, top1_w, diff


def _collect_rhos(f_mask, w_mask, n_take: int, rhos: list, cap: int) -> None:
    """Append up to n_take per-position Spearman rhos until the global cap is hit."""
    if len(rhos) >= cap:
        return
    f_cpu = f_mask.cpu().numpy()
    w_cpu = w_mask.cpu().numpy()
    for i in range(min(n_take, f_cpu.shape[0])):
        if len(rhos) >= cap:
            break
        r = spearman_rho(f_cpu[i], w_cpu[i])
        if r is not None:
            rhos.append(r)


def run_characterization(
    adapter: ModelAdapter,
    weak_selfs: Dict[str, WeakSelf],
    cfg,
    arrays_dir: str = None,
) -> Dict[str, str]:
    """Run the two-pass characterization for each construction and dump its arrays.

    Args:
        adapter: full model adapter (same model used for both full and weak passes).
        weak_selfs: mapping {construction_name: WeakSelf}.
        cfg: Phase0Config.
        arrays_dir: where to dump npz files (defaults to cfg.arrays_dir).

    Returns:
        dict {construction_name: npz_path} for every construction successfully dumped.
    """
    from autoguidance.phase0.arrays import save_arrays

    if arrays_dir is None:
        arrays_dir = getattr(cfg, "arrays_dir", "phase0_arrays")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = random.Random(cfg.seed)

    texts = _load_dataset_texts(cfg, cfg.n_samples)
    supports_embed = getattr(adapter, "supports_embed_noise", True)

    print("=" * 70)
    print("[Phase 0] run_characterization")
    print(f"  device={adapter.device} dtype={getattr(cfg, 'dtype', '?')} "
          f"precision={getattr(cfg, 'precision', '?')}")
    print(f"  n_samples={len(texts)} mask_rate={cfg.mask_rate} "
          f"vocab={adapter.vocab_size} mask_id={adapter.mask_token_id}")
    print(f"  constructions={list(weak_selfs.keys())}")
    print(f"  arrays_dir={arrays_dir}")
    print(f"  supports_embed_noise={supports_embed}")
    print("=" * 70)

    # Spearman is expensive over a ~126k vocab; cap total sampled positions and
    # spread the budget evenly across samples (not biased to the first texts).
    spearman_cap = 2000
    per_sample_rho = max(1, spearman_cap // max(1, cfg.n_samples))

    paths: Dict[str, str] = {}

    for name, weak_self in weak_selfs.items():
        print(f"\n[Phase 0] === construction: {name} ===")

        if name in _EMBED_NOISE_CONSTRUCTIONS and not supports_embed:
            print(f"  [skip] {name} needs embedding-noise support "
                  f"(adapter.supports_embed_noise=False)")
            continue

        # Per-masked-position reduced quantities only — never [N, L, vocab].
        ent_full, ent_weak = [], []
        full_top1, weak_top1 = [], []
        diff_mag, gt_flat = [], []
        rhos: list = []
        n_processed = 0

        try:
            from tqdm import tqdm
            iter_texts = tqdm(texts, desc=f"  {name}", unit="sample")
        except ImportError:
            iter_texts = texts

        for si, text in enumerate(iter_texts):
            try:
                token_ids = adapter.encode(text)    # [1, L]
                if token_ids.shape[1] < 4:
                    continue

                gt_ids, mask_ind = _mask_positions(token_ids, cfg.mask_rate, rng)
                gt_dev = gt_ids.to(adapter.device)
                mask_dev = mask_ind.to(adapter.device)
                masked_ids = adapter.corrupt_positions(gt_dev, mask_dev)  # [1, L]
                m = mask_ind.squeeze(0)             # [L] bool, CPU

                with torch.no_grad():
                    f_logits = adapter.logits(masked_ids)[0]            # [L, vocab]
                    w_logits = weak_self(masked_ids, None, adapter)[0]  # [L, vocab]

                    mdev = m.to(f_logits.device)
                    f_mask = f_logits[mdev].float()    # [n_masked, vocab]
                    w_mask = w_logits[mdev].float()

                ef, ew, ft, wt, dm = _reduce_position_arrays(f_mask, w_mask)
                ent_full.append(ef)
                ent_weak.append(ew)
                full_top1.append(ft)
                weak_top1.append(wt)
                diff_mag.append(dm)
                gt_flat.append(gt_ids.squeeze(0).cpu().numpy()[m.numpy()])

                _collect_rhos(f_mask, w_mask, per_sample_rho, rhos, spearman_cap)

                del f_logits, w_logits, f_mask, w_mask
                n_processed += 1

                if (si + 1) % 25 == 0:
                    print(f"    [{name}] {si + 1}/{len(texts)} processed "
                          f"(masked positions so far={sum(len(a) for a in ent_full)}, "
                          f"rho_samples={len(rhos)})")

            except Exception as e:
                print(f"    [{name}] skipping sample {si}: {e}")
                continue

        if n_processed == 0:
            print(f"  [{name}] ERROR: no samples processed — not dumping arrays")
            continue

        ef = np.concatenate(ent_full)
        ew = np.concatenate(ent_weak)
        ft = np.concatenate(full_top1)
        wt = np.concatenate(weak_top1)
        dm = np.concatenate(diff_mag)
        gt = np.concatenate(gt_flat)
        rho_sample = np.asarray(rhos, dtype=np.float32)

        meta = {
            "construction": name,
            "n_samples_processed": n_processed,
            "n_masked_positions": int(ef.shape[0]),
            "mask_rate": cfg.mask_rate,
            "seed": cfg.seed,
            "model": type(adapter).__name__,
            "vocab_size": int(adapter.vocab_size),
            "max_seq_len": getattr(cfg, "max_seq_len", None),
            "spearman_cap": spearman_cap,
            "per_sample_rho": per_sample_rho,
            "n_rho_samples": int(rho_sample.shape[0]),
        }
        path = save_arrays(
            arrays_dir, name, meta,
            entropy_full=ef, entropy_weak=ew,
            top1_full=ft, top1_weak=wt,
            diff_mag=dm, gt=gt, rho_sample=rho_sample,
        )
        paths[name] = path
        print(f"  [{name}] dumped {ef.shape[0]} masked positions "
              f"({n_processed} samples, {rho_sample.shape[0]} rho samples)")

    print(f"\n[Phase 0] characterization complete — dumped {len(paths)} construction(s): "
          f"{list(paths.keys())}")
    return paths

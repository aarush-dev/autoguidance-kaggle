"""Phase 0 array I/O.

Each weak-self construction's per-masked-position reduced quantities are dumped to
a single compressed ``.npz`` file under ``arrays_dir``. Thresholding is a separate,
model-free step (thresholds.py) that reloads these arrays — so the expensive forward
passes run once and the cheap pass/fail criteria can be re-evaluated or swept freely.

File layout (one per construction):
    <arrays_dir>/<name>.npz
        meta         — JSON string (dict) describing how the arrays were produced
        entropy_full — [n_masked] per-position Shannon entropy of the full model
        entropy_weak — [n_masked] per-position Shannon entropy of the weak self
        top1_full    — [n_masked] argmax token id, full model
        top1_weak    — [n_masked] argmax token id, weak self
        diff_mag     — [n_masked] max_v |full_logit - weak_logit|
        gt           — [n_masked] ground-truth token id at each masked position
        rho_sample   — [<=cap] sampled per-position Spearman rho (full vs weak logits)

Never store the raw [N, L, vocab] logits — only these reduced arrays cross to disk
(see CLAUDE.md memory rule).
"""
from __future__ import annotations
from typing import Dict, List
import glob
import json
import os

import numpy as np


def save_arrays(arrays_dir: str, name: str, meta: dict, **named_arrays) -> str:
    """Dump one construction's reduced arrays to ``<arrays_dir>/<name>.npz``.

    Args:
        arrays_dir: output directory (created if missing).
        name: construction name; the file is ``<name>.npz``.
        meta: JSON-serializable dict, stored as a single json string array entry.
        **named_arrays: the reduced numpy arrays (entropy_full, top1_full, ...).

    Returns:
        Absolute path to the written ``.npz`` file.
    """
    os.makedirs(arrays_dir, exist_ok=True)
    if name.endswith(".npz"):
        name = name[: -len(".npz")]
    path = os.path.join(arrays_dir, f"{name}.npz")

    meta_json = json.dumps(meta, default=_json_default)
    payload = {k: np.asarray(v) for k, v in named_arrays.items()}
    payload["meta"] = np.array(meta_json)  # 0-d string array

    np.savez_compressed(path, **payload)
    abs = os.path.abspath(path)
    print(
        f"[arrays] saved {name}: {abs} "
        f"({', '.join(f'{k}={tuple(np.asarray(v).shape)}' for k, v in named_arrays.items())})"
    )
    return abs


def load_arrays(arrays_dir: str, name: str) -> Dict:
    """Load one construction's arrays. ``meta`` is returned as a parsed dict."""
    if name.endswith(".npz"):
        name = name[: -len(".npz")]
    path = os.path.join(arrays_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[arrays] no array file at {path}")

    out: Dict = {}
    with np.load(path, allow_pickle=False) as data:
        for k in data.files:
            if k == "meta":
                out["meta"] = json.loads(str(data["meta"]))
            else:
                out[k] = data[k]
    return out


def list_constructions(arrays_dir: str) -> List[str]:
    """Return the construction names (npz stems) present in ``arrays_dir`` (top level)."""
    if not os.path.isdir(arrays_dir):
        return []
    files = sorted(glob.glob(os.path.join(arrays_dir, "*.npz")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

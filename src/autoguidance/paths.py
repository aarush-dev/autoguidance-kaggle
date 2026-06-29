"""Filesystem + offline-mode helpers for Kaggle-style runs.

Everything here is deliberately verbose: a human watches stdout, so each helper
prints what it resolved / extracted / set.

Typical flow on Kaggle (no internet):
    from autoguidance.paths import set_hf_offline, resolve_mount, ensure_extracted
    set_hf_offline()                              # BEFORE importing transformers
    mount = resolve_mount(kaggle_cfg, kaggle_cfg.slug_llada)
    model_dir = ensure_extracted(mount, kaggle_cfg.local_dir(kaggle_cfg.slug_llada))
"""
from __future__ import annotations

import os
import tarfile


def set_hf_offline() -> None:
    """Force the HuggingFace stack fully offline.

    MUST be called before importing transformers / datasets so the env vars are
    read at import time. Idempotent.
    """
    flags = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for k, v in flags.items():
        os.environ[k] = v
    print(f"[paths] HF offline mode set: {', '.join(f'{k}={v}' for k, v in flags.items())}")


def resolve_mount(kaggle_cfg, slug: str) -> str:
    """Resolve the read-only mount path for an attached dataset slug.

    Kaggle mounts a dataset 'owner/dataset-name' at '<input_root>/dataset-name'.
    """
    name = slug.rstrip("/").split("/")[-1]
    path = os.path.join(kaggle_cfg.input_root, name)
    exists = os.path.isdir(path)
    print(f"[paths] resolve_mount slug={slug!r} -> {path} (exists={exists})")
    if not exists:
        print(f"[paths] WARNING: mount {path} does not exist. "
              f"Is the dataset attached? input_root={kaggle_cfg.input_root}")
    return path


def ensure_extracted(tar_or_dir: str, dest: str) -> str:
    """Return a usable directory for a dataset payload.

    - If tar_or_dir is already a directory, it is returned as-is (nothing to do).
    - If it is a .tar / .tar.gz / .tgz file, it is extracted into dest (once;
      skipped if dest already exists and is non-empty) and dest is returned.
    """
    if os.path.isdir(tar_or_dir):
        print(f"[paths] ensure_extracted: {tar_or_dir} is a directory; using directly.")
        return tar_or_dir

    if not os.path.isfile(tar_or_dir):
        raise FileNotFoundError(f"[paths] ensure_extracted: not a dir or file: {tar_or_dir!r}")

    if os.path.isdir(dest) and os.listdir(dest):
        print(f"[paths] ensure_extracted: dest {dest} already populated; skip extract.")
        return dest

    os.makedirs(dest, exist_ok=True)
    print(f"[paths] ensure_extracted: extracting {tar_or_dir} -> {dest} ...")
    with tarfile.open(tar_or_dir) as tf:
        members = tf.getmembers()
        tf.extractall(dest)
    print(f"[paths] ensure_extracted: done ({len(members)} members) -> {dest}")
    return dest

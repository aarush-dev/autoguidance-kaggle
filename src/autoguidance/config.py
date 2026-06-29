from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import os
import yaml


@dataclass
class BaseConfig:
    seed: int = 42
    device_main: str = "cuda:0"
    device_eval: str = "cuda:0"
    device_cpu: str = "cpu"
    precision: str = "bf16"          # single full-bf16 path on cuda:0
    dtype: str = "bfloat16"
    batch_size: int = 8
    max_seq_len: int = 256
    # Selects which prebuilt wheels dataset to use:
    #   "tf5"   -> transformers 5.x stack (gemma)
    #   "tf446" -> transformers 4.46 stack (llada)
    transformers_variant: str = "tf5"


@dataclass
class KaggleConfig:
    """Kaggle filesystem layout + dataset slugs.

    On Kaggle, attached datasets mount read-only under input_root at a directory
    named after the slug's final path component (the owner prefix is dropped).
    Anything we need to write/extract goes under temp_root or work_root.
    """
    input_root: str = "/kaggle/input"
    temp_root: str = "/kaggle/temp"
    work_root: str = "/kaggle/working"

    slug_llada: str = "aarush-dev/autoguidance-llada-8b"
    slug_diffusiongemma: str = "aarush-dev/autoguidance-diffusiongemma"
    slug_gpt2_scorer: str = "aarush-dev/autoguidance-gpt2-large"
    slug_mauve: str = "aarush-dev/autoguidance-mauve"
    slug_wheels: str = "aarush-dev/autoguidance-wheels"
    slug_hfdata: str = "aarush-dev/autoguidance-hfdata"
    slug_code: str = "aarush-dev/autoguidance-code"

    def local_dir(self, slug: str) -> str:
        """Extracted/usable temp path for a dataset slug.

        Returns a path under temp_root named after the slug's final component,
        which is where ensure_extracted() lands a tarball (or where a plain
        mounted directory can be symlinked/copied for a uniform interface).
        """
        name = slug.rstrip("/").split("/")[-1]
        return os.path.join(self.temp_root, name)


@dataclass
class Phase0Config(BaseConfig):
    n_samples: int = 200
    mask_rate: float = 0.5
    dataset: str = "wikitext"
    dataset_split: str = "validation"
    constructions: List[str] = field(
        default_factory=lambda: [
            "input_noise_remask",
            "input_noise_gauss",
            "low_nfe",
            "layer_drop",
            "inference_dropout",
            "reduced_expert",
        ]
    )
    # Construction params
    noise_rate_remask: float = 0.20
    noise_sigma_gauss: float = 0.1
    low_nfe_steps: int = 1
    layer_drop_k: int = 2
    dropout_p: float = 0.1
    reduced_expert_topk: int = 4
    # Parameter sweeps (re-run weak pass only)
    sweep_remask_rate: List[float] = field(
        default_factory=lambda: [0.2, 0.25, 0.3, 0.35, 0.4]
    )
    sweep_gauss_sigma: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5]
    )
    sweep_layer_drop_k: List[int] = field(default_factory=lambda: [1, 2, 4])
    sweep_dropout_p: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    # Pass thresholds
    check1_entropy_fraction: float = 0.80
    check2_top1_agreement: float = 0.60
    check2_spearman_rho: float = 0.50
    check3_pearson_r: float = 0.20
    check3_precision: float = 0.50
    # Where characterization arrays are dumped
    arrays_dir: str = "/kaggle/working/phase0_arrays"


@dataclass
class Phase1Config(BaseConfig):
    n_samples: int = 100
    nfe_ladder: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])
    decoders: List[str] = field(default_factory=lambda: ["unguided", "cfg", "acfg"])
    cfg_weight: float = 2.0
    acfg_threshold: float = 0.3
    dataset: str = "wikitext"
    dataset_split: str = "validation"
    sudoku_n_puzzles: int = 100
    self_bleu_n_pairs: int = 100
    mauve_max_text: int = 1000
    scorer_model: str = "gpt2-large"


def load_config(path: str, cls):
    with open(path) as f:
        data = yaml.safe_load(f)
    # drop unknown keys so dataclass doesn't choke
    valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    return cls(**valid)

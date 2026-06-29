# Public Interfaces

## ModelAdapter (`src/autoguidance/models/base.py`)

```python
class ModelAdapter(ABC):
    def logits(self, x_t: LongTensor, cond: Optional[dict] = None) -> FloatTensor
        # [B, seq, vocab] -- single forward pass
    def embed(self, x_t: LongTensor) -> FloatTensor
        # [B, seq, d_model] -- input embeddings
    def logits_from_embed(self, embeds: FloatTensor, cond: Optional[dict] = None) -> FloatTensor
        # [B, seq, vocab] -- forward from embeddings (input_noise_gauss)
    def corrupt_positions(self, ids: LongTensor, mask: BoolTensor, generator=None) -> LongTensor
        # replace masked positions per this model's noise process
        # LLaDA: ids[mask] = mask_token_id
    @property
    def supports_embed_noise(self) -> bool
        # gates input_noise_gauss; default True
    @property
    def n_layers(self) -> int
        # total transformer layers; used by LayerDropWeakSelf
    @property
    def is_moe(self) -> bool
        # True if model has MoE routing; gates reduced_expert; default False
    def _decoder_layers(self) -> nn.ModuleList
        # hook returning the transformer layer list (for layer-drop)
    @property
    def mask_token_id(self) -> int
    @property
    def vocab_size(self) -> int
    @property
    def device(self) -> torch.device
    def encode(self, text: str) -> LongTensor      # [1, seq]
    def decode(self, token_ids: LongTensor) -> str
```

Implementations: `SyntheticAdapter` (CPU, tests), `LLaDAAdapter` (bf16, cuda:0), `DiffusionGemmaAdapter` (bf16, cuda:0; load API unverified — confirm at build time).
Stub: `DreamAdapter`.

LLaDA extra method:
```python
def logits_cfg(self, x_t, prompt_mask, cfg_scale) -> FloatTensor
    # CFG via batch concat; used by CFGSampler
```

## WeakSelf (`src/autoguidance/weak_self/base.py`)

```python
class WeakSelf(ABC):
    def __call__(self, x_t: LongTensor, cond: Optional[dict], adapter: ModelAdapter) -> FloatTensor
        # [B, seq, vocab] -- weak logits
```

Implementations:
- `InputNoiseRemask(remask_rate)` — input_noise_remask: re-mask a fraction of unmasked tokens
- `InputNoiseGauss(sigma)` — input_noise_gauss: Gaussian noise on embeddings; raises NotImplementedError if not adapter.supports_embed_noise
- `LowNFEWeakSelf(nfe)` — low_nfe: single-step prediction from fully-masked input
- `LayerDropWeakSelf(k)` — layer_drop: drop last k transformer layers during weak pass
- `InferenceDropoutWeakSelf(dropout_p, seed)` — inference_dropout: enable dropout during weak pass
- `ReducedExpertWeakSelf(topk)` — reduced_expert: restrict MoE to top-k experts; raises NotImplementedError if not adapter.is_moe

Factory: `load_weak_self(name, cfg)` in `weak_self/__init__.py` maps each construction name to the correct class with the correct config parameters.

## Phase 0 arrays (`src/autoguidance/phase0/arrays.py`)

```python
def save_arrays(arrays_dir: str, name: str, meta: dict, **named_arrays) -> str
    # saves named_arrays to arrays_dir/<name>.npz (compressed); meta stored as json string entry
    # named_arrays: entropy_full, entropy_weak, top1_full, top1_weak, diff_mag, gt, rho_sample

def load_arrays(arrays_dir: str, name: str) -> dict
    # loads and returns all arrays + parsed meta dict

def list_constructions(arrays_dir: str) -> list[str]
    # returns list of construction names with saved arrays in arrays_dir
```

## Phase 0 thresholds (`src/autoguidance/phase0/thresholds.py`)

```python
def evaluate_arrays(arrays: dict, cfg: Phase0Config) -> dict
    # applies check1/check2/check3 via eval/* modules; returns {check1, check2, check3, all_pass}

def evaluate_dir(arrays_dir: str, cfg: Phase0Config) -> dict[str, dict]
    # loads all saved arrays and calls evaluate_arrays per construction; returns {name -> verdict}
```

## Phase 0 sweep (`src/autoguidance/phase0/sweep.py`)

```python
def sweep_threshold(arrays_dir: str, grid: dict, cfg: Phase0Config) -> list[dict]
    # sweeps threshold values over saved arrays; pure numpy, no model
    # returns list of rows with columns: construction, threshold_name, threshold_value, check1, check2, check3, all_pass

def sweep_construction_param(
    adapter: ModelAdapter,
    cfg: Phase0Config,
    param_name: str,
    values: list,
    arrays_dir: str,
    full_cache: Optional[dict] = None,
) -> dict[Any, str]
    # re-runs only the weak-self pass for each param value; reuses cached full-pass arrays
    # full_cache keyed by (model_id, seed, mask_rate) to avoid recomputing full pass
    # returns {value -> npz_path}
```

## SamplerBase (`src/autoguidance/samplers/base.py`)

```python
class SamplerBase(ABC):
    def sample(self, prompts: List[str], nfe: int) -> List[str]
```

Implementations: `UnguidedSampler`, `CFGSampler`, `ACFGSampler`.
Stub: `GuidedSampler` (Phase 2).

Shared utility:
```python
def run_denoising_loop(adapter, input_ids, nfe, get_logits, temperature, seed, acfg_threshold=None) -> LongTensor
```

## Config dataclasses (`src/autoguidance/config.py`)

`BaseConfig`, `KaggleConfig`, `Phase0Config`, `Phase1Config`. Load from YAML with `load_config(path, cls)` (drops unknown keys silently).

See CONFIG CONTRACT in CLAUDE.md for full field listings and defaults.

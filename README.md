# autoguidance

Discrete masked-text autoguidance — Phase 0 and Phase 1.

**Phase 0:** Characterize weak-self constructions across six variants (input noise remask, input noise Gaussian, low-NFE, layer drop, inference dropout, reduced expert). Confirm they are "more uncertain, not differently wrong." Characterization dumps raw per-position arrays to disk; threshold evaluation and sweeps read those arrays without touching the model again.

**Phase 1:** Baseline evaluation — unguided, CFG, A-CFG across the NFE ladder {4, 8, 16, 32, 64, 128, 256} on LLaDA-8B, producing the quality-diversity Pareto frontier.

Hardware: single RTX PRO 6000 (96 GB), cuda:0, bf16 throughout.

---

## Install (local development)

```bash
cd build/autoguidance
pip install -r requirements.txt
pip install -e .
```

---

## CPU smoke test (no GPU, under 1 minute)

```bash
# All unit tests on synthetic model
PYTHONPATH=src pytest tests/ -v --tb=short

# Phase 0 smoke test on synthetic model (dumps arrays to /tmp/phase0_arrays_smoke)
python scripts/run_phase0.py --model synthetic --n-samples 10

# Phase 1 smoke test on synthetic model
python scripts/run_phase1.py --model synthetic --n-samples 5 --nfe 4,8
```

---

## Real experiment (GPU required)

### Phase 0 — Weak-self characterization

Phase 0 runs in two steps. **Do not re-run the characterize step to change a threshold.**

#### Step 1: Characterize (GPU, model loaded, ~2-4 hours)

```bash
# bf16 on cuda:0 (default)
python scripts/run_phase0.py --model llada

# DiffusionGemma (confirm load API at build time before using this)
python scripts/run_phase0.py --model diffusiongemma
```

Arrays land in `outputs/phase0_arrays/` (one `.npz` per construction).

#### Step 2: Evaluate thresholds and sweeps (CPU only, seconds)

```bash
# Re-evaluate pass/fail from saved arrays (no model needed)
python scripts/run_phase0.py --eval-only --arrays-dir outputs/phase0_arrays

# Sweep remask rate without re-running the full model
python scripts/run_phase0.py --sweep remask_rate --arrays-dir outputs/phase0_arrays
```

Results land in `outputs/phase0/`. Check `outputs/phase0/verdict.txt` for the pass/fail table.

**Do not start Phase 1 until Phase 0 reports at least one construction passing all 3 checks.**

#### How to make Phase 0 pass

If no construction passes out of the box:
1. Run a threshold sweep: inspect the printed sweep table and lower failing thresholds in `configs/phase0.yaml` if the construction is close.
2. Run a construction-parameter sweep (`--sweep gauss_sigma`, `--sweep layer_drop_k`, etc.) to find a better operating point. Only the cheap weak-self pass is re-run; the full-model arrays are reused from disk.
3. Update thresholds in `configs/phase0.yaml` and re-run `--eval-only`. The model is never re-loaded for threshold changes.

### Phase 1 — Baselines (~6-12 hours on RTX PRO 6000)

```bash
python scripts/run_phase1.py --model llada
```

Results land in `outputs/phase1/`. Key files:
- `outputs/phase1/results.csv` — one row per (decoder, NFE)
- `outputs/phase1/pareto.png` — quality-diversity Pareto plot

---

## Kaggle two-notebook flow

All production runs execute on Kaggle with internet disabled in the runner. Two notebooks:

### Builder notebook (internet ON, run once)

1. `pip install` packages from PyPI.
2. Download LLaDA-8B (or DiffusionGemma) weights from HuggingFace hub.
3. Package `src/` as a `.tar.gz` and upload as a Kaggle dataset (`slug_code`).
4. Upload weights, wheel files (per `transformers_variant`: `tf5` or `tf446`), and data as separate Kaggle datasets.

### Runner notebook (internet OFF)

1. Attach all datasets: weights, wheels, data, code.
2. Set env: `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`.
3. Install wheels from local mount path (no PyPI).
4. Unpack code dataset, prepend to `PYTHONPATH`.
5. Run Phase 0 or Phase 1 script pointing at local dataset paths via `KaggleConfig`.

`KaggleConfig.local_dir(slug)` resolves each slug to its Kaggle mount path. Pass a `KaggleConfig` instance to override `input_root` and slug fields if paths differ.

### Wheel variant selection

`BaseConfig.transformers_variant` controls which wheels dataset to install:
- `"tf5"` — use `slug_wheels` pointing at the tf5-compatible wheels (for DiffusionGemma / Gemma backbone).
- `"tf446"` — use `slug_wheels` pointing at the tf446-compatible wheels (for LLaDA-8B).

Set in `configs/base.yaml` before building.

---

## Code structure

```
src/autoguidance/
  config.py               - dataclass configs (BaseConfig, KaggleConfig, Phase0Config, Phase1Config)
  models/
    base.py               - ModelAdapter ABC (logits, embed, logits_from_embed, corrupt_positions, ...)
    synthetic.py          - tiny CPU model (for tests)
    llada.py              - LLaDA-8B adapter (bf16, cuda:0)
    diffusiongemma.py     - DiffusionGemma adapter (bf16, cuda:0; verify load API at build time)
    dream.py              - STUB (Phase 2)
  weak_self/
    base.py               - WeakSelf ABC
    input_noise.py        - input_noise_remask + input_noise_gauss (Construction A)
    low_nfe.py            - low_nfe (Construction B)
    layer_drop.py         - layer_drop (Construction C)
    inference_dropout.py  - inference_dropout (Construction D)
    reduced_expert.py     - reduced_expert (Construction E, DiffusionGemma MoE)
  eval/
    entropy.py            - entropy_stats_from_arrays
    agreement.py          - agreement_stats_from_arrays
    error_position.py     - error_position_stats_from_arrays
    gen_ppl.py, mauve.py, ...
  samplers/
    base.py               - SamplerBase + run_denoising_loop
    unguided.py           - plain masked diffusion
    cfg_sampler.py        - CFG (LLaDA formula)
    acfg_sampler.py       - A-CFG (re-mask low confidence)
    guided_step.py        - STUB (Phase 2)
  phase0/
    characterize.py       - run_characterization -> dumps .npz arrays, no pass/fail
    arrays.py             - save_arrays / load_arrays / list_constructions
    thresholds.py         - evaluate_arrays / evaluate_dir (pure numpy, no model)
    sweep.py              - sweep_threshold / sweep_construction_param
    report.py             - render verdict + plots from arrays
  phase1/
    baseline_runner.py    - run_baselines
    report.py             - save_results
scripts/
  run_phase0.py           - Phase 0 CLI (--eval-only, --sweep, --arrays-dir flags)
  run_phase1.py           - Phase 1 CLI
configs/
  base.yaml               - BaseConfig defaults
  phase0.yaml             - Phase0Config (all 6 constructions, sweep grids, arrays_dir)
  phase1.yaml             - Phase1Config
tests/                    - all tests run on synthetic model (no GPU)
outputs/
  phase0_arrays/          - generated .npz files (characterize step)
  phase0/                 - rendered reports (threshold/sweep step)
  phase1/                 - baseline results
docs/
  phase0.md               - Phase 0 design notes
  phase1.md               - Phase 1 design notes
  interfaces.md           - public Python interfaces
```

---

## Deferred (Phase 2+)

- `samplers/guided_step.py`: the autoguidance step `full + w*(full-weak)` with w-sweep
- `models/dream.py`: Dream-7B adapter (Phase 2)

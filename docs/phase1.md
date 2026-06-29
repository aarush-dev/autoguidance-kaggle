# Phase 1: Baselines

## Goal

Establish the quality-diversity Pareto frontier that autoguidance must later beat.

## Decoders

| Decoder | Description |
|---------|-------------|
| unguided | Plain masked diffusion, no guidance |
| cfg | Built-in CFG using LLaDA formula (batch-concat conditional + unconditional) |
| acfg | A-CFG: re-mask tokens below confidence threshold tau=0.3 at each step (arXiv:2505.20199) |

## NFE ladder

{4, 8, 16, 32, 64, 128, 256}

## Metrics (per decoder x NFE)

| Metric | Device | Notes |
|--------|--------|-------|
| gen_ppl | cuda:0 | GPT-2-large, bf16 |
| task_acc | CPU | Sudoku accuracy |
| distinct1 | CPU | Unique unigram fraction |
| distinct2 | CPU | Unique bigram fraction |
| self_bleu | CPU | Lower = more diverse |
| mauve | cuda:0 | vs WikiText-103 reference, bf16 |

All eval runs on the single RTX PRO 6000 (cuda:0) or CPU — no second GPU.

## Precision

All inference uses `bfloat16` on `cuda:0`. No quantization, no offloading, no fp16.

## Running

```bash
# Smoke test (no GPU, 5 samples, NFE=4,8)
python scripts/run_phase1.py --model synthetic --n-samples 5 --nfe 4,8

# Real run (RTX PRO 6000, bf16, ~6-12 hours)
python scripts/run_phase1.py --model llada

# DiffusionGemma (confirm load API at build time before using)
python scripts/run_phase1.py --model diffusiongemma
```

## Outputs

- `outputs/phase1/results.csv` — one row per (decoder, NFE)
- `outputs/phase1/pareto.png` — MAUVE (x) vs gen_ppl (y), marker size proportional to NFE

## Acceptance

All 3 decoders x 7 NFE values complete. CSV written deterministically (same seed). Pareto plot shows at least one quality-diversity trade-off trend.

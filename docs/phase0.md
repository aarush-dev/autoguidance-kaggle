# Phase 0: Weak-Self Characterization

## Goal

Before any guidance, confirm that each weak-self construction satisfies the "same errors, amplified" precondition. If it does not, guidance degenerates like CFG-with-a-broken-reference.

## Three checks

| # | Check | Pass criterion |
|---|-------|---------------|
| 1 | Entropy | mean(H_weak) > mean(H_full) AND >80% of positions have H_weak > H_full |
| 2 | Top-1 agreement | top1_agreement >= 0.60 AND mean Spearman rho >= 0.50 |
| 3 | Error-position correlation | Pearson r >= 0.20 OR precision >= 0.50 |

## Constructions

Six constructions are evaluated:

| Name | Group | Description |
|------|-------|-------------|
| `input_noise_remask` | A | Re-mask 20% of unmasked tokens before weak forward pass |
| `input_noise_gauss` | A | Gaussian noise (sigma=0.1) on input embeddings; gated by `adapter.supports_embed_noise` |
| `low_nfe` | B | Single-step prediction (NFE=1) from fully-masked input |
| `layer_drop` | C | Drop the last k=2 transformer layers during weak forward pass |
| `inference_dropout` | D | Enable dropout (p=0.1) during weak forward pass |
| `reduced_expert` | E | Restrict MoE routing to top-k=4 experts; requires DiffusionGemma (gated by `adapter.is_moe`) |

Constructions that require an unsupported capability (e.g. `reduced_expert` on LLaDA, `input_noise_gauss` on an adapter without embed support) raise `NotImplementedError`; `run_characterization` catches and skips them with a logged warning.

## Architecture: arrays / thresholds / sweep split

Phase 0 has two distinct steps that must not be conflated:

### Step 1: Characterize (expensive — GPU, model loaded)

`characterize.run_characterization(adapter, weak_selfs, cfg, arrays_dir)` runs two forward passes per sample:
1. Full model forward pass.
2. Each enabled weak-self forward pass.

Per sample, reduce to 7 named arrays and discard logits immediately:
- `entropy_full` — per-position Shannon entropy from full logits [N, L_mask]
- `entropy_weak` — per-position entropy from weak logits [N, L_mask]
- `top1_full` — top-1 token id from full logits [N, L_mask]
- `top1_weak` — top-1 token id from weak logits [N, L_mask]
- `diff_mag` — max absolute logit difference [N, L_mask]
- `gt` — ground-truth token ids at masked positions [N, L_mask]
- `rho_sample` — per-sample Spearman rho (scalar) [N]

Arrays are saved via `arrays.save_arrays(arrays_dir, name, meta, **named_arrays)` as `.npz` compressed files with a JSON metadata entry. One file per construction.

Use `adapter.corrupt_positions(ids, mask)` to apply each construction's noise process (LLaDA: replace with mask token id). Gate `input_noise_gauss` on `adapter.supports_embed_noise`. Gate `reduced_expert` on `adapter.is_moe`.

### Step 2: Threshold evaluation (cheap — pure numpy, no model)

`thresholds.evaluate_arrays(arrays, cfg)` and `thresholds.evaluate_dir(arrays_dir, cfg)` read the saved `.npz` files and apply check1/check2/check3 via the existing `eval/` modules:
- `eval.entropy.entropy_stats_from_arrays`
- `eval.agreement.agreement_stats_from_arrays`
- `eval.error_position.error_position_stats_from_arrays`

Returns a dict with `check1`, `check2`, `check3`, `all_pass` verdict per construction.

**Rule: change a threshold by editing `configs/phase0.yaml` and re-running `thresholds.evaluate_dir`. Never re-run `run_characterization` to change a threshold.**

**Data-quality guards (added after the mixed-model results.zip incident):**
- **One model per arrays_dir.** `evaluate_dir` hard-fails if the `.npz` files carry more than one `meta["model"]`. Arrays from different models are not comparable and must never share a verdict table. The runner notebook namespaces outputs per model (`phase0_arrays/<MODEL>/`) so two model runs cannot collide in one dir.
- **Degenerate-entropy gate.** If the full model's median per-position entropy is `< 1e-3` nats, Check 1 ("weak is more uncertain") cannot discriminate and is forced to FAIL with a `degenerate_full_entropy` flag. This is the signature of position misalignment — e.g. DiffusionGemma's block-diffusion canvas output being cropped/padded to the input length reads *known/context* slots (near one-hot) instead of the masked positions. Treat such a run as invalid, not as a real negative result, until the adapter's per-position alignment is verified on real weights.

### Step 3: Sweeps (cheap — re-runs weak pass only, full arrays reused from disk)

`sweep.sweep_threshold(arrays_dir, grid, cfg)` sweeps threshold values over saved arrays — pure numpy, no model.

`sweep.sweep_construction_param(adapter, cfg, param_name, values, arrays_dir, full_cache)` re-runs only the weak-self pass for each new parameter value. The full-pass arrays are cached per `(model, seed, mask_rate)` and never recomputed.

## Running

```bash
# Smoke test (< 1 min, no GPU)
python scripts/run_phase0.py --model synthetic --n-samples 10

# Real run — bf16 on cuda:0 (RTX PRO 6000 96GB, ~2-4 hours)
python scripts/run_phase0.py --model llada

# DiffusionGemma — verify load API at build time before using
python scripts/run_phase0.py --model diffusiongemma

# Threshold evaluation only (no model, reads saved arrays)
python scripts/run_phase0.py --eval-only --arrays-dir outputs/phase0_arrays

# Sweep construction parameter (re-runs only weak pass)
python scripts/run_phase0.py --sweep remask_rate --arrays-dir outputs/phase0_arrays
```

## Outputs

- `outputs/phase0_arrays/<name>.npz` — raw arrays per construction (characterize step)
- `outputs/phase0/results.json` — structured pass/fail results (threshold step)
- `outputs/phase0/agreement_table.md` — pass/fail table
- `outputs/phase0/entropy_dist.png` — per-position entropy distributions
- `outputs/phase0/error_position_scatter.png` — disagreement vs. error correlation
- `outputs/phase0/verdict.txt` — which constructions pass, recommendation

## Memory: statistics are streamed

`characterize.py` reduces each sample to per-masked-position quantities (entropy, top-1 ids, max logit-diff magnitude, ground truth) inside the sample loop and discards the `[L, vocab]` logits immediately. It never accumulates `[N, L, vocab]` tensors — at 200 x 256 x 126k vocab x bf16 that is approximately 13 GB per tensor per construction, far exceeding safe usage. Per-position scalars are streamed to CPU and accumulated as numpy arrays. Spearman keeps a 2000-position cap spread evenly across samples (`per_sample_rho`).

## Precision

All forward passes use `bfloat16` on `cuda:0`. The full model and each weak-self always use the same precision. No fp16, no quantization, no offloading.

## DiffusionGemma and reduced_expert

`reduced_expert` gates on `adapter.is_moe` — it only runs if the adapter reports a MoE architecture. DiffusionGemma is the expected model for this construction. The DiffusionGemma load API is unverified as of project start; confirm the HuggingFace class and weight layout at build time before implementing `DiffusionGemmaAdapter`.

## Acceptance

At least one construction passes all 3 checks. If none pass: run sweeps, adjust thresholds, re-evaluate. If still none pass after sweeps: stop, report negative result, do not proceed to Phase 1.

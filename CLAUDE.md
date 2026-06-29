# CLAUDE.md — Operating Rules for autoguidance

Every agent spawned for this project reads this file first.

## Project

Discrete-text autoguidance research. Port Karras-style autoguidance (weak-self logit difference) to masked diffusion LMs. Phase 0 and Phase 1 only; all else is stubbed.

## Model routing

| Task | Agent model |
|------|-------------|
| Reading papers, summarizing, studying repos | sonnet |
| Writing code, architecture decisions | opus |
| Shell commands, git, pip install, test runs | haiku |

## Parallel dynamic workflow policy

Decompose each phase into independent units and run concurrently. Fan out understanding across sonnet agents (one per doc cluster), coding across opus agents (one per module group), shell work through haiku agents — all in parallel. Join results before the next phase. Never run a heavier model than the task needs.

Use workflows (multi-agent orchestration) for debugging, investigation, and any task that spans multiple files or requires parallel diagnosis. Do not debug inline in the main conversation when a workflow can parallelize the work.

## YAGNI decision process

Before writing anything beyond the current MVP need, ask: is this required for Phase 0 or Phase 1 to pass its smoke test right now? If not, stub it behind an interface. Any rewrite of working code requires a one-paragraph justification in PLAN.md; if it cannot be justified, do not rewrite.

## `repositories/` convention

If a YAGNI decision concludes that an external repo is needed, clone it into `repositories/<repo-name>/`. Write `repositories/<repo-name>/UNDERSTANDING.md` capturing the architecture, which parts matter here, and how to reuse them. Every future agent touching that repo reads UNDERSTANDING.md first.

## Per-phase documentation discipline

After each phase completes: write or update `docs/phase0.md` or `docs/phase1.md`. After every code change: update the affected doc in the same change. Stale docs are a bug.

## Git authorship rules

- Author and commit only as `aarush-dev`.
- `git config user.name "aarush-dev"` and `git config user.email "aarush-dev@users.noreply.github.com"` must be set before any commit.
- No Claude, Anthropic, or assistant names in author, committer, blame, or commit messages.
- No `Co-Authored-By` trailers. No `Generated with` notes. The blame shows only `aarush-dev`.

## Hardware profile

| Resource | Precision |
|----------|-----------|
| CUDA:0 — RTX PRO 6000 (96 GB, Ada Lovelace) | bf16 — single GPU, all workloads |

- **Single GPU only.** All inference, scoring, and MAUVE run on `cuda:0`.
- Default precision: `bfloat16`. No fp16, no 4bit, no 8bit, no fp16_offload, no bitsandbytes.
- DiffusionGemma (26B) fits in 96 GB bf16. Adapter must be confirmed at build time — the load API is unverified; confirm which HuggingFace class and weight layout to use before writing `DiffusionGemmaAdapter`.
- No WSL2 CUDA allocator workarounds required; this hardware does not exhibit that limitation.
- Never accumulate full `[N, L, vocab]` logit tensors across samples — at 200 samples × 256 seq × 126k vocab × bf16 this exceeds 10 GB per tensor and inflates peak usage unnecessarily. Stream statistics per-sample: reduce to per-position scalars/ids on-GPU, move to CPU, discard logits immediately.
- Full model and weak self always use the same precision.
- Device, precision, batch size: always from config, never hardcoded.
- `torch.no_grad()` everywhere during inference.

## Kaggle offline execution rules

The Kaggle runtime runs fully offline. All weights, HuggingFace wheels, data, and project code arrive as attached Kaggle datasets — GitHub is never cloned at runtime, and `pip install` from the internet is not available.

Two-notebook split:
- **Builder notebook** (internet ON): installs packages, downloads weights and data, packages the code as a Kaggle dataset. Runs once, produces datasets.
- **Runner notebook** (internet OFF): attaches all datasets, sets `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, loads everything from local dataset mount paths, runs Phase 0 or Phase 1 end to end.

Key rules:
- `KaggleConfig.local_dir(slug)` returns the extracted path for each attached dataset slug.
- Wheels are selected per `transformers_variant`: `"tf5"` attaches the gemma-compatible wheels dataset; `"tf446"` attaches the llada-compatible wheels dataset. The runner installs from the local wheels path.
- HuggingFace `from_pretrained` always receives `local_files_only=True` and the dataset mount path; never a hub model ID at runtime.
- Code ships as a dataset (a tarball or zip of `src/`); the runner unpacks it and prepends to `PYTHONPATH`.

## `transformers_variant` — wheel selection

`BaseConfig.transformers_variant` (`"tf5"` or `"tf446"`) selects which wheels dataset to attach and install:

| Value | Transformers version | Primary use |
|-------|---------------------|-------------|
| `tf5` | 5.x (latest, e.g. 4.50+) | DiffusionGemma (Gemma backbone) |
| `tf446` | 4.46.x | LLaDA-8B (requires 4.46 API) |

The runner reads `cfg.transformers_variant` to pick the correct dataset slug before installing wheels.

## Phase 0 architecture: arrays/thresholds/sweep split

Phase 0 has two distinct steps that must not be conflated:

1. **Characterize step** (`phase0/characterize.py`): runs both the full model and each weak-self construction forward pass over N samples, reduces each sample to 7 named arrays (`entropy_full`, `entropy_weak`, `top1_full`, `top1_weak`, `diff_mag`, `gt`, `rho_sample`), and dumps them to disk via `save_arrays(arrays_dir, name, ...)`. This step is **expensive** (GPU, model loaded) and **must not be re-run to change a threshold**.

2. **Threshold/sweep step** (`phase0/thresholds.py`, `phase0/sweep.py`): reads the saved `.npz` arrays, applies the check1/check2/check3 criteria, and reports pass/fail. Sweeps re-read the arrays and optionally re-run only the weak-self pass (never the full pass) for new parameter values. Pure numpy — no torch model required.

Rule: if a threshold changes, re-run thresholds.py, not characterize.py. If a sweep is needed, re-run sweep.py using the cached full-pass arrays.

## DiffusionGemma load API

The HuggingFace class and weight layout for DiffusionGemma are unverified as of project start. Before implementing `DiffusionGemmaAdapter`:
- Confirm at build time which class loads the weights (`AutoModelForMaskedLM`, `AutoModel`, or a custom class).
- Confirm the mask token id and vocab size.
- Write a minimal load smoke-test in the builder notebook before writing adapter code.

## Deferred phases (do not implement)

- Phase 2: `guided_step.py` — the `full + w*(full - weak)` guided arithmetic and w-sweep.
  Stub: `raise NotImplementedError("Phase 2: implement guided_step after Phase 1 is green.")`
- Dream adapter: `raise NotImplementedError("Phase 2: Dream adapter not wired.")`

## Scope

MVP = Phase 0 + Phase 1. Stop here. Report each phase green before touching the next.

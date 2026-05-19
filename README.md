# Geometry-Adaptive Explainer (GAE): Code

This repository contains the implementation, evaluation pipeline, and reproduction
scripts for **Geometry-Adaptive Explainer (GAE)**, a training-free post-hoc method
that restores explanation faithfulness of dictionary-based explainers under
distribution shift.

Setting-specific reproduction notes live in
`markdowns/{Domain-OOD,Timeshift-OOD,Adversarial-OOD}.md` and the method note in
`markdowns/GAE.md`.

---

## 1) Scope

This repository evaluates explanation faithfulness of mechanistic explainers under
distribution shift.

Covered dimensions:
- **Models:** GPT-2 Small, Pythia-1.4B (frozen pretrained backbones)
- **Explainers:** Transcoder, Top-K SAE (both with dictionary size `k = 32d`)
- **OOD settings:**
  - Temporal — FineWeb
  - Domain — Edgar (financial filings)
  - Adversarial — HaluEval
- **Baselines:** Fixed, TERM, Finetune, Retrain, SAEBoost, FaithfulSAE, **GAE (ours)**

---

## 2) Codebase

Core implementation:
- `run_experiment.py` — main runner (CLI, baseline dispatch, evaluation)
- `gae.py` — GAE algorithm (Step 1 Procrustes alignment, Step 2 closed-form decoder)
- `train_explainers.py` — ID-explainer (Transcoder / Top-K SAE) training
- `saeboost.py` — SAEBoost residual booster
- `ood_utils/` — dataset loaders, evaluation, diagnostics
- `sae_training/` — SAELens-based training utilities

Backbones are loaded via TransformerLens (`HookedTransformer.from_pretrained`).
Default mid-layer mapping (`config.py`): GPT-2 `L=8`, Pythia-1.4B `L=15`.
Evaluation is performed at the last token (`pos=-1`).

---

## 3) Explainers and Baselines

ID-trained explainers (`--explainer transcoder|sae`) use `k = 32d`:
- GPT-2 Small (`d=768`): `DICT_SIZE_K = 24576`
- Pythia-1.4B (`d=2048`): `DICT_SIZE_K = 65536`

Baselines (`--baselines ...`):
| Name | Type | Description |
|---|---|---|
| `fixed` | training-free | ID explainer applied as-is on OOD |
| `term` | training-based (ID) | Tilted ERM training on ID data |
| `ood_finetuned` | training-based (OOD) | Warm-start the ID explainer on OOD activations (≈ Finetune) |
| `ood_retrained` | training-based (OOD) | Train explainer from scratch on OOD activations (≈ Retrain) |
| `saeboost` | training-based (OOD) | Residual booster on top of the ID explainer |
| `faithfulsae` | training-time | Re-train explainer on the model's own generations |
| `gae` (ours) | **training-free** | Geometric adaptation from unlabeled OOD activations |

GAE in two steps (paper Section 4):
1. **Step 1 — Procrustes alignment.** Align the explainer's reference subspace to
   the OOD-active subspace, estimated as the top-`r` eigenspace of the OOD
   covariance.
2. **Step 2 — Closed-form decoder.** Within the aligned subspace, fit the decoder
   in closed form on unlabeled OOD activations. No gradient updates.

---

## 4) Evaluation Metrics (paper)

We report three causal-faithfulness metrics:
- **nAOPC** (Normalized AOPC, ↑): averaged normalized logit drop across feature
  budgets when the top-`m` features are removed.
- **nComp** (Normalized comprehensiveness, ↑): normalized logit drop at a single
  budget `m* = 32`.
- **|ΔCE|** (Delta cross-entropy, ↓ to zero): cross-entropy change when hidden
  activations are replaced with the explainer's reconstruction.

Implementation lives in `ood_utils/evaluation.py`. Defaults:
`M = [1, 2, 4, 8, 16, 32, 64, 128]`, `m* = 32`, `empty_mode = zero_resid`.

---

## 5) Shared Run Protocol

For each `(model, explainer, ood_set, baseline)` combination, the runner:
1. Loads the frozen backbone and the ID-trained explainer checkpoint.
2. Builds an OOD activation pool from OOD train-side data.
3. Extracts OOD activations / logits at the fixed layer at `pos=-1`.
4. Adapts the explainer per the selected baseline (training-free for `gae`,
   gradient-based for `ood_retrained`, `ood_finetuned`, `saeboost`).
5. Evaluates causal faithfulness (nAOPC / nComp / |ΔCE|) on a held-out OOD eval
   set.
6. Saves a JSON summary to `results/`.

Default scale and constraints (`config.py`):
- `MAX_LEN = 256`, `MIN_LEN = 64`
- `N_OOD_DOMAIN = 20000`, `N_COV ≈ 2048` activations for OOD covariance
- `seed = 2026`

---

## 6) CLI and Output

Entry points:
- `python run_experiment.py timeshift --model {gpt2,pythia-1.4b} --explainer {transcoder,sae} --checkpoint PATH --baselines ... --ood_set fineweb`
- `python run_experiment.py domain    --model ... --explainer ... --checkpoint PATH --baselines ... --ood_set edgar`
- `python run_experiment.py adv       --model ... --explainer ... --checkpoint PATH --baselines ... --ood_set halu_eval`

Common flags:
`--baselines fixed term ood_retrained ood_finetuned saeboost faithfulsae gae`,
`--n_eval`, `--target_mode {argmax,gold,provided}`, `--faith_m`, `--faith_m_star`.

Result file naming:
- Temporal:    `results/timeshift_ood_{model}_{explainer}_{ood_set}_{baseline_str}.json`
- Domain:      `results/domain_ood_{model}_{explainer}_{ood_set}_{baseline_str}.json`
- Adversarial: `results/adversarial_ood_{model}_{explainer}_{ood_set}_{baseline_str}.json`

---

## 7) Reproduction Scripts

Reproduction shell scripts cover the three OOD settings × seven baselines used in
the paper:

```
scripts/
├── train_id/         # ID explainer training (Transcoder, Top-K SAE, TERM, FaithfulSAE)
├── timeshift_ood/    # FineWeb (Temporal)
├── domain_ood/       # Edgar (Domain)
└── adv_ood/          # HaluEval (Adversarial)
```

Each script takes positional args, e.g.:
```bash
bash scripts/timeshift_ood/gae.sh <DEVICE> <MODEL>     # MODEL ∈ {gpt2, pythia-1.4b}
```

Before running, set the following environment variables:
- `REPO_ROOT`  — repository root (used for `cd` in scripts)
- `REPO_DATA`  — directory holding checkpoints (`${REPO_DATA}/checkpoints/...`)
- `DATA_ROOT`  — HF / dataset cache root (`${DATA_ROOT}/hf_cache`, `${DATA_ROOT}/datasets_cache`)

---

## 8) Setting-Specific Notes

- **Temporal (FineWeb):** see `markdowns/Timeshift-OOD.md`. OOD set: `fineweb`.
- **Domain (Edgar):** see `markdowns/Domain-OOD.md`. OOD set: `edgar`. Uses a
  JSONL prompt cache built by `build_domain_ood_prompts.py`.
- **Adversarial (HaluEval):** see `markdowns/Adversarial-OOD.md`. OOD set:
  `halu_eval` (QA + Dialogue subsets).

---

## 9) Reproducibility Checklist

Before each run:
- confirm the checkpoint type matches `--explainer`
- confirm `--n_eval`, `--faith_m`, `--faith_m_star` are held fixed across baselines
- record `gae_r` and any retraining hyperparameters

After each run:
- verify result file naming matches the expected task prefix
- check that every baseline block contains all three causal metrics
- compare runs only when non-data settings are identical

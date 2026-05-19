# Domain OOD (Companion Note)

This file keeps only Domain-specific deltas.
For full experimental protocol, metrics, adaptation rules, and CLI contracts, read `README.md` first.

Primary reference in `README.md`:
- `## 9) Domain OOD (Detailed Setting Spec)`
- `## 6) Shared Adaptation and Evaluation Protocol`
- `## 7) Global Metrics (Implemented)`
- `## 8) CLI and Output Contracts`

---

## 1) Domain Shift Definition

Domain OOD evaluates source-level distribution shift caused by domain/discourse changes.
Compared to Time-shift OOD, it introduces stronger structural and terminology changes.

---

## 2) Supported OOD Sets

- `patents`
- `edgar`

---

## 3) Data Path (What Changes vs Other Settings)

### 3.1 JSONL-first path (default)

- Runner first tries `{domain_ood_jsonl_dir}/{ood_set}_{model}.jsonl`.
- If file exists and non-empty, JSONL records drive activation extraction and evaluation token construction.
- Optional target override: `--domain_ood_force_provided_target`.

### 3.2 HF fallback path

Fallback is used when:
- JSONL missing/empty, or
- `--no_domain_ood_jsonl` is set.

Then the runner uses:
- `ood_utils/data_domain.load_ood_dataset`
- prompt construction via `create_factual_prompts`

---

## 4) Domain JSONL Generation (If Needed)

Generator:
- `build_domain_ood_prompts.py`

Typical script:
- `scripts/domain_ood/make_prompts.sh`

Generated files:
- `patents_{model}.jsonl`
- `edgar_{model}.jsonl`

---

## 5) Run Scripts

- `scripts/domain_ood/fixed.sh`
- `scripts/domain_ood/term.sh`
- `scripts/domain_ood/gae.sh`
- `scripts/domain_ood/ood_retrained.sh`

---

## 6) Domain-only Flags to Check

- `--domain_ood_jsonl_dir`
- `--no_domain_ood_jsonl`
- `--no_domain_ood_jsonl_streaming`
- `--domain_ood_force_provided_target`

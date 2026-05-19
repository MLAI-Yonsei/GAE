# Time-shift OOD (Companion Note)

This file keeps only Time-shift-specific deltas.
For full experimental protocol, metrics, adaptation rules, and CLI contracts, read `README.md` first.

Primary reference in `README.md`:
- `## 10) Time-shift OOD (Detailed Setting Spec)`
- `## 6) Shared Adaptation and Evaluation Protocol`
- `## 7) Global Metrics (Implemented)`
- `## 8) CLI and Output Contracts`

---

## 1) Time-shift Definition

Time-shift OOD isolates temporal drift while keeping broad web domain similar to ID.
Compared to Domain OOD, structural shift is usually smaller.

---

## 2) Supported OOD Sets

- `fineweb`
- `dolma_web`

---

## 3) Data Path (What Changes vs Other Settings)

### 3.1 FineWeb

- source: `HuggingFaceFW/fineweb`
- streaming loading
- text field chosen from `text/content/document/data`

### 3.2 Dolma Web

- local JSON/JSONL file pool under `DATA_DIR`
- optional `DOLMA_FILE_GLOB` to restrict files
- accepted formats include `json.gz`, `jsonl.gz`, `jsonl.zst`, `jsonl`

Helper:
- `scripts/timeshift_ood/download_dolma_subset.sh`

---

## 4) Prompt and Target Policy

Prompt path:
- generic `create_factual_prompts` from OOD test split

Target policy:
- default `target_mode='argmax'`
- no Domain-style provided-target JSONL path in this setting

Implementation note:
- activation sampling uses `span_mode='middle'` for `pythia-410m`
- others use `span_mode='prefix'`

---

## 5) Run Scripts

- `scripts/timeshift_ood/fixed.sh`
- `scripts/timeshift_ood/term.sh`
- `scripts/timeshift_ood/gae.sh`
- `scripts/timeshift_ood/ood_retrained.sh`

---

## 6) Expected Pattern

- temporal/topic drift without major domain-style change
- `fixed` and `term` can still degrade in faithfulness
- `gae` improves geometric and causal metrics over non-adaptive baselines

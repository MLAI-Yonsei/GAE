# Adversarial OOD (Companion Note)

This file keeps only Adversarial-specific deltas.
For full experimental protocol, metrics, adaptation rules, and CLI contracts, read `README.md` first.

Primary reference in `README.md`:
- `## 11) Adversarial OOD (Detailed Setting Spec)`
- `## 6) Shared Adaptation and Evaluation Protocol`
- `## 7) Global Metrics (Implemented)`
- `## 8) CLI and Output Contracts`

Adversarial OOD evaluates explanation faithfulness under **failure-inducing input distributions** that systematically stress the model’s internal reliance structure.

Unlike Domain or Time-shift OOD, Adversarial OOD is defined behaviorally:

> Inputs that induce hallucination or safety override, potentially altering hidden-space activation geometry and feature reliance patterns.

Adversarial OOD in this work consists of:

- Hallucination-Adversarial OOD (HaluEval)
- Behavior-Adversarial OOD (JailbreakBench)
- Structural Jailbreak OOD (JailbreakHub) — **new, replaces JailbreakBench as primary jailbreak benchmark**

TruthfulQA is excluded to maintain a strict adversarial definition.

### JailbreakBench → JailbreakHub Migration

JailbreakBench (`JailbreakBench/JBB-Behaviors`) contains only 100 behaviors (32 test samples after 70/30 split), which is too small for reliable metric estimation. We replace it with JailbreakHub (`walledai/JailbreakHub`) which provides 1,405 in-the-wild jailbreak prompts (428 test samples after split).

Key differences:
- JailbreakBench: curated behavior catalog (100 entries)
- JailbreakHub: real jailbreak prompts from Reddit, Discord, FlowGPT (1,405 entries)
- Both contain structural adversarial prompts (role-play, DAN, prompt injection, instruction override)
- JailbreakHub published at ACM CCS 2024 (Shen et al., arXiv:2308.03825)

JailbreakBench results are retained for reference but JailbreakHub is the primary jailbreak evaluation set.

---

# 1. Taxonomy of Adversarial OOD

## 1.1 Hallucination-Adversarial OOD

Dataset:
- `pminervini/HaluEval`

Definition:
Prompts designed to induce factual hallucination or unreliable reasoning.

This setting tests whether explanation faithfulness degrades
when the model relies on unstable internal representations.

---

## 1.2 Behavior-Adversarial OOD

Dataset:
- `JailbreakBench/JBB-Behaviors`

Definition:
Prompts designed to override safety or alignment constraints.

This setting tests whether explanation faithfulness degrades
under behavioral override pressure.

---

# 2. Dataset Usage Strategy

We enforce strict separation between:

1. OOD activation pool (for geometry estimation)
2. OOD evaluation set (for faithfulness evaluation)

Principle:

- Geometry adaptation is **label-free**
- Evaluation may be **label-aware**

---

# 3. Hallucination OOD (HaluEval)

## 3.1 Subset Selection

HaluEval includes:

- QA
- Dialogue
- Summarization

Final decision:

We use:

- QA
- Dialogue

We exclude:

- Summarization (long context, strong decoding artifacts, misaligned with pos=-1 analysis)

---

## 3.2 OOD Activation Pool (for GAE / OOD-Retrained / SAEBoost)

We construct a unified hallucination activation pool:

- Combine QA + Dialogue
- Ignore labels
- Accumulate samples until token budget reached

Token-budget policy:

```

N_OOD_TOKENS = fixed constant (e.g., 2M)

```

Stop collecting once token budget is reached.

This pool is used for:

- GAE geometry estimation
- OOD-retrained dictionary learning
- SAEBoost adaptation (if applicable)

No hallucination labels are used in adaptation.

---

## 3.3 Evaluation Construction (Hallucination)

### QA Evaluation

Prompt template:

```

Question: {query}
Answer:

```

Columns used:

- `query`
- `response`
- `label`

Target modes:

- `provided` (preferred if gold available)
- `argmax`
- correctness-conditioned reporting

Evaluation breakdown:

- overall
- hallucinated-only
- non-hallucinated-only

---

### Dialogue Evaluation

Prompt template:

```

{dialogue history}
Assistant:

```

Columns used:

- `dialogue`
- `response`
- `label`

Same reporting structure:

- overall
- hallucinated-only
- non-hallucinated-only

---

## 3.4 Target Policy and Reporting Strategy (Hallucination)

Supported:

- `target_mode='provided'`
- `target_mode='argmax'`

Main tables:
- provided (if available)

Supplementary:
- argmax

Additionally report:

- Faithfulness on hallucinated-only subset
- Faithfulness on non-hallucinated subset

This prevents self-justifying explanation effects.

---

## 3.5 Short-Prompt Stability Strategy (Hallucination)

HaluEval QA may include short prompts.

We enforce:

### (1) Token-budget-based OOD pool (mandatory)
Use total token budget instead of sample count.

### (2) Bootstrap confidence intervals (recommended)
Bootstrap over evaluation samples.

### (3) Anchor position (optional)
If needed:
- first token after `Answer:` for QA
- first token after assistant turn for Dialogue

Default remains `pos=-1`.

---

# 4. Jailbreak OOD (JailbreakBench)

## 4.1 Dataset Structure

JailbreakBench includes:

- ~100 harmful behaviors
- Multiple prompt variants per behavior

Dataset size is relatively small.

---

## 4.2 OOD Activation Pool Construction

To ensure stable covariance estimation:

- Pool all behaviors
- Pool all prompt variants
- Ignore labels
- Apply token-budget policy

Split strategy:

- 70% prompts → activation pool
- 30% prompts → evaluation

Split at prompt level.

---

## 4.3 Evaluation Construction (Jailbreak)

Prompt template:

```

Instruction: {prompt}
Response:

```

Columns used:

- `behavior`
- `prompt`
- `response`

Target modes:

- `argmax`
- optional refusal-aware analysis

Evaluation reporting:

- overall faithfulness
- behavior-level average
- bootstrap over behaviors (recommended)

---

## 4.4 Target Policy and Reporting Strategy (Jailbreak)

Primary mode:

- `argmax`

Optional:
- refusal-conditioned reporting
- behavior-level aggregation

Because jailbreak data lacks canonical "gold" answers,
argmax is treated as response-faithfulness evaluation.

---

## 4.5 Short-Prompt Stability Strategy (Jailbreak)

Jailbreak prompts can be short.

We enforce:

### (1) Token-budget-based activation pool (mandatory)

### (2) Behavior-level bootstrap (mandatory)

### (3) Optional anchor position:
first token after `Response:`

---

# 4b. JailbreakHub (Primary Jailbreak Benchmark)

## 4b.1 Dataset

- HuggingFace: `walledai/JailbreakHub`
- Paper: Shen et al., "Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models", ACM CCS 2024 (arXiv:2308.03825)
- Size: 15,140 total prompts, 1,405 after `jailbreak == True` filter
- Source: In-the-wild collection from Reddit, Discord, FlowGPT, jailbreak communities
- License: CC-BY-NC-4.0

## 4b.2 Why JailbreakHub over JailbreakBench

| | JailbreakBench | JailbreakHub |
|---|---|---|
| Total samples | 100 behaviors | 1,405 jailbreak prompts |
| Test samples (70/30 split) | 32 | 428 |
| Source | Curated behavior catalog | In-the-wild collection |
| Structural attack patterns | Limited (behavior descriptions) | 79.7% explicit structural attacks |
| Publication | arXiv | ACM CCS 2024 |

## 4b.3 Dataset Structure

Fields:
- `prompt`: The jailbreak prompt text
- `jailbreak`: Boolean flag (True = jailbreak prompt)
- `platform`: Source platform (reddit, discord, etc.)
- `source`: Specific community/channel

We filter to `jailbreak == True` and apply the same deterministic hash-based train/test split.

## 4b.4 Prompt Template

```
Instruction: {prompt}
Response:
```

Same template as JailbreakBench. The `prompt` field is used directly.

## 4b.5 OOD Activation Pool

- Pool all jailbreak prompts from train split (~977 samples)
- Apply token-budget policy
- No labels used

## 4b.6 Evaluation

- Test split: ~428 samples
- Target mode: `argmax`
- Bootstrap over prompts (recommended)
- Reports: overall faithfulness metrics (nAOPC, nSuff, nComp)

## 4b.7 CLI Usage

```bash
python run_experiment.py adv --model MODEL --explainer transcoder --ood_set jailbreakhub ...
```

Environment variables:
- `JAILBREAKHUB_PROMPT_SPLIT_TRAIN_RATIO`: Train/test split ratio (default: 0.7)
- `JAILBREAKHUB_DATA`: Local data path fallback

---

# 5. Measuring Hidden-Space Shift Before Main Experiments

We empirically validate adversarial shift before running faithfulness experiments.

Metrics:

1. RAER
2. Angle alignment
3. Energy-weighted alignment

Energy-weighted alignment:

$$
\mathrm{Align}_{\Sigma} = \frac{ \mathrm{tr}(U_{ID}^\top \Sigma_{OOD} U_{ID})}{\mathrm{tr}(\Sigma_{OOD})}
$$

We compute separately for:

- Hallucination OOD
- Jailbreak OOD

Shift evidence:

- RAER ↑
- Energy alignment ↓

---

# 6. Baseline Usage

## 6.1 GAE

- Uses unified OOD activation pool
- No labels
- Separate evaluation per dataset

---

## 6.2 OOD-Retrained

- Retrains dictionary using OOD activation pool
- Backbone frozen
- No labels used

---

## 6.3 SAEBoost / Fixed

- ID-trained dictionary
- No OOD adaptation
- Evaluated on hallucination and jailbreak separately

---

# 7. Interpretation Notes

Adversarial OOD may induce:

- Subspace rotation
- Feature re-weighting
- Behavioral override without full rotation

Angle-based metrics capture rotation.
Energy-weighted alignment captures both rotation and re-weighting.

Both are reported.
## SAEBoost Baseline Design for TC

This document specifies how to add a **SAEBoost-style baseline** to the TC repository, for both **SAE** and **Transcoder (MLP-out only)** explainers, without depending on the external `SAEBoost-main` code.

The goal is to make it easy to:

- **Train residual models** on OOD activations (residual SAE / residual Transcoder)
- **Wrap base + residual** into a single explainer interface
- **Evaluate** with the existing TC metrics (geometric + causal faithfulness)

---

## 1. Terminology and Scope

- **Explainers** (fixed by this repo):
  - **`SAE`**: SAELens sparse autoencoder on hidden states
  - **`Transcoder` (MLP-out)**: SAELens transcoder whose `output_kind="mlp_out"` (maps input-hook hidden \(\rightarrow\) MLP-out hidden, e.g. `ln2.hook_normalized -> hook_mlp_out`)
- **Baselines** (per explainer):
  - **`fixed`**: ID-trained explainer, evaluated on OOD as-is (existing baseline)
  - **`term`**: TERM-trained checkpoint (existing)
  - **`gae`**: GAE geometric adaptation (existing)
  - **`ood_retrained`**: explainer fully retrained on OOD activations (existing)
  - **`saeboost` (new)**: base explainer + residual model trained on **residuals** for each OOD setting

Constraints:

- Only **MLP-out Transcoder** is considered for SAEBoost.
- **Residual Transcoder** and **residual SAE** are trained **separately per OOD setting / OOD set**, not shared across all OODs.
- We **do not use** the `SAEBoost-main` code as a library; it is only a conceptual reference.

---

## 2. High-Level SAEBoost Idea in TC

For both SAE and Transcoder, SAEBoost follows the same additive pattern:

- **SAE case (single hook space)**:
  - Let $h \in \mathbb{R}^{d}$ be the hidden activation at the SAE hook.
  - Base SAE reconstruction: $\hat{h}_{\text{base}}$
  - Residual target: $r = h - \hat{h}_{\text{base}}$
  - Final reconstruction: $\hat{h}_{\text{boost}} = \hat{h}_{\text{base}} + \hat{r}$
- **Transcoder case (input/output hook pair)**:
  - Let $x_{\text{in}} \in \mathbb{R}^{d}$ be activation at `input_hook_point` and $y_{\text{out}} \in \mathbb{R}^{d}$ be activation at `output_hook_point`.
  - Base Transcoder predicts $\hat{y}_{\text{base}} = f_{\text{base}}(x_{\text{in}})$.
  - Residual target is defined in output space:
    $$
    r = y_{\text{out}} - \hat{y}_{\text{base}}
    $$
  - Residual model predicts $\hat{r}$ from $x_{\text{in}}$.
  - Final output reconstruction: $\hat{y}_{\text{boost}} = \hat{y}_{\text{base}} + \hat{r}$
- **Final features** (both cases):
  $$
  z_{\text{boost}} = [z_{\text{base}}, z_{\text{resid}}]
  $$

This lets us:

- Keep the **ID geometry and semantics** of the base explainer
- Add **OOD-specific capacity** via a small residual model
- Reuse the existing **TC metrics** by exposing SAEBoost as a single explainer.

---

## 3. SAEBoost for SAE Explainers

### 3.1 Base SAE

- **Checkpoint**: existing SAELens SAE, trained on ID activations.
- **Interface** (already in TC via SAELens adapter):
  - `base_explainer(h, return_z=True) -> (h_base, z_base)`
  - `base_explainer.decoder(z_base) -> h_base`
  - Dictionary $B_{\text{base}} \in \mathbb{R}^{d \times k_{\text{base}}}$ accessible via `W_dec` or `saelens.W_dec`.
- Base SAE is **frozen** in all SAEBoost runs.

### 3.2 Residual SAE (per OOD setting / set)

**Data and target**

- Use the existing OOD activation pipeline:
  - Sample OOD train text for a given task (domain / timeshift / adv)
  - Extract activations $h$ at the same hook as the base SAE (e.g. `resid_post`, layer L).
- For each batch:
  1. Compute base reconstruction: $(h_{\text{base}}, z_{\text{base}}) = \text{base explainer}(h, \text{return_z=True})$
  2. Define residual target: $r = h - h_{\text{base}}$
- Residual SAE:
  - Input: $h$
  - Target: $r$
  - Loss: reconstruction loss on $r$ (MSE + sparsity regularization).

**Architecture & hyperparameters**

- **Dictionary size**:
  - Smaller than base, e.g. $k_{\text{resid}} \ll k_{\text{base}}$
  - Rule of thumb: `dict_size_resid ≈ 1/4 ~ 1/8` of base dict size.
- **Sparsity**:
  - Same family as base (e.g. BatchTopK / TopK / L1) but typically **smaller top-k**.
- **Tokens / steps**:
  - Residual SAE sees **many fewer tokens** than the base SAE (OOD pool only).
  - Concrete values should be chosen to roughly match SAEBoost-main’s “residual SAE” scale (e.g. tens of millions of tokens, not billions).

### 3.3 SAEBoostExplainerAdapter (SAE)

Implement a wrapper that exposes **base SAE + residual SAE** as a single explainer:

- Internals:
  - `self.base_explainer` (ID SAE, frozen)
  - `self.resid_explainer` (residual SAE)

- Forward:
  - Given hidden \(h\):
    - $(h_{\text{base}}, z_{\text{base}}) = \text{base explainer}(h, \text{return_z=True})$
    - $(h_{\text{resid}}, z_{\text{resid}}) = \text{resid explainer}(h, \text{return_z=True})$
    - $z_{\text{boost}} = [z_{\text{base}}, z_{\text{resid}}]$
    - $\hat{h}_{\text{boost}} = h_{\text{base}} + h_{\text{resid}}$
  - Return `(h_boost, z_boost)` when `return_z=True`.

- Decoder and dictionary:
  - Let $B_{\text{base}} \in \mathbb{R}^{d \times k_{\text{base}}}$, $B_{\text{resid}} \in \mathbb{R}^{d \times k_{\text{resid}}}$.
  - Define **combined dictionary**
    $$
    B_{\text{boost}} = [B_{\text{base}}, B_{\text{resid}}] \in \mathbb{R}^{d \times (k_{\text{base}} + k_{\text{resid}})}
    $$
  - For decoding:
    - Split $z_{\text{boost}}$ into $z_{\text{base}}$ and $z_{\text{resid}}$
    - Decode each with its own decoder and sum:
      $$
      \hat{h}_{\text{boost}} = D_{\text{base}} z_{\text{base}} + D_{\text{resid}} z_{\text{resid}}
      $$
  - Expose `W_dec = B_boost.T` so existing geometric metrics (RAER, alignment) work without change.

- Hook metadata:
  - Reuse `input_hook_point` / `output_hook_point` from the base SAE.

This adapter should be registered in `ood_utils.saelens_adapters.EXPLAINER_TYPES` so that `evaluate_causal_faithfulness` and related evaluation paths treat it as a standard explainer.

---

## 4. SAEBoost for Transcoder (MLP-out)

### 4.1 Base Transcoder (MLP-out only)

- We restrict to **MLP-out Transcoder**:
  - `output_kind="mlp_out"`
  - The decoder reconstructs **hidden activations** at the output hook (MLP-out stream), not logits.
- Interface (via SAELens transcoder adapter):
  - `base_transcoder(x_in, return_z=True) -> (y_base, z_base)`
  - `base_transcoder.decoder(z_base) -> y_base`
  - Input is taken from `input_hook_point` (typically `ln2.hook_normalized`), and output/reconstruction is in `output_hook_point` (typically `hook_mlp_out`).
  - Dictionary $B_{\text{base}} \in \mathbb{R}^{d \times k_{\text{base}}}$ spans the output (MLP-out) space.
- Base Transcoder is an **ID-trained `fixed` explainer** and remains frozen in SAEBoost.

### 4.2 Residual Transcoder (per OOD setting / set)

Instead of a residual SAE, we train a **residual Transcoder** with its own (smaller) dictionary and encoder/decoder, following the same residual target definition.

**Data and target**

- Use paired OOD activations:
  - \(x_{\text{in}}\): at transcoder `input_hook_point`
  - \(y_{\text{out}}\): at transcoder `output_hook_point` (MLP-out)
- For each batch:
  1. Base Transcoder pass:
     $$
     (\hat{y}_{\text{base}}, z_{\text{base}}) = \text{base transcoder}(x_{\text{in}}, \text{return_z=True})
     $$
  2. Residual target:
     $$
     r = y_{\text{out}} - \hat{y}_{\text{base}}
     $$

**Residual Transcoder model**

- A **Transcoder-style model** with:
  - Own encoder mapping $x_{\text{in}} \rightarrow z_{\text{resid}} \in \mathbb{R}^{k_{\text{resid}}}$
  - Own decoder mapping $z_{\text{resid}} \rightarrow \hat{r}$ in output (MLP-out) space
  - No logits head: `output_kind="mlp_out"` only.
- Loss:
  - Reconstruction loss on residuals:
    $$
    \mathcal{L}_{\text{resid}} = \|r - \hat{r}\|^2 + \text{sparsity_penalty}(z_{\text{resid}})
    $$

**Hyperparameters**

- **Dict size**:
  - $k_{\text{resid}} \ll k_{\text{base}}$, residual Transcoder is significantly smaller.
- **Sparsity / architecture**:
  - Follow the **Transcoder training style in this repo** but scaled down:
    - Same kind of encoder/decoder structure
    - Smaller hidden sizes / bottleneck width as needed
  - Tokens: only OOD activations; total token count similar to residual SAE case.

### 4.3 SAEBoostTranscoderAdapter

Analogous to the SAE case, implement a wrapper that combines the **base Transcoder + residual Transcoder** into a single explainer.

- Internals:
  - `self.base_transcoder`
  - `self.resid_transcoder`

- Forward:
  - Given transcoder input hidden \(x_{\text{in}}\):
    - $(\hat{y}_{\text{base}}, z_{\text{base}}) = \text{base transcoder}(x_{\text{in}}, \text{return_z=True})$
    - $(\hat{r}, z_{\text{resid}}) = \text{resid transcoder}(x_{\text{in}}, \text{return_z=True})$
    - $z_{\text{boost}} = [z_{\text{base}}, z_{\text{resid}}]$
    - $\hat{y}_{\text{boost}} = \hat{y}_{\text{base}} + \hat{r}$
  - Return `(y_boost, z_boost)` when `return_z=True`.

- Decoder and dictionary:
  - Let $B_{\text{base}} \in \mathbb{R}^{d \times k_{\text{base}}}$ and $B_{\text{resid}} \in \mathbb{R}^{d \times k_{\text{resid}}}$ be the dictionaries for base and residual Transcoder decoders.
  - Combined dictionary:
    $$
    B_{\text{boost}} = [B_{\text{base}}, B_{\text{resid}}]
    $$
  - Decoding:
    - Split $z_{\text{boost}}$ into base/resid parts
    - Decode each with its own decoder and sum to obtain $\hat{h}_{\text{boost}}$.
  - Expose `W_dec = B_boost.T` (for geometric metrics).

- Hook metadata:
  - Reuse `input_hook_point` / `output_hook_point` from the base Transcoder.
  - This allows existing code that uses hook names (e.g. `forward_with_override_logits`) to treat SAEBoostTranscoder as a drop-in replacement.

---

## 5. Training & Checkpoint Conventions

### 5.1 Checkpoint naming (suggested)

For clarity and automation, adopt the following pattern (example):

- Base explainer (already exists):
  - `checkpoints/{task}/{model}_{explainer}_id.pt`
- Residual model (new, per OOD set):
  - `checkpoints/{task}/{model}_{explainer}_{ood_set}_saeboost_resid.pt`
    - `explainer ∈ {sae, transcoder}`
    - `task ∈ {domain_ood, timeshift_ood, adversarial_ood}`

This makes it easy for the runner to:

- Load the **ID base explainer** for a given `(model, explainer)`
- Load the **residual model** for a given `(task, ood_set, explainer)`

### 5.2 CLI baseline name

- Extend `--baselines` to accept **`saeboost`**.
- Filenames for results follow existing conventions:
  - Domain: `results/domain_ood_{model}_{explainer}_{ood_set}_saeboost.json`
  - Timeshift: `results/timeshift_ood_{model}_{explainer}_{ood_set}_saeboost.json`
  - Adversarial: `results/adversarial_ood_{model}_{explainer}_{ood_set}_saeboost.json`

---

## 6. Runner Integration (`run_experiment.py`)

At a high level, the runner should do the following for the **`saeboost`** baseline:

- **1) Load base explainer (`fixed`, ID-trained)**:
  - Same loader as for `fixed`/`gae`/`ood_retrained`, but always the **ID** checkpoint (no GAE / OOD retrain applied to the base).

- **2) Load residual model checkpoint**:
  - Use the suggested naming convention to locate:
    - Residual SAE (for `explainer='sae'`)
    - Residual Transcoder (for `explainer='transcoder'`)

- **3) Construct SAEBoost adapter**:
  - If `explainer == 'sae'`: use `SAEBoostExplainerAdapter`
  - If `explainer == 'transcoder'`: use `SAEBoostTranscoderAdapter`
- **3.5) Register SAEBoost adapters as explainer types**:
  - Add `SAEBoostExplainerAdapter` and `SAEBoostTranscoderAdapter` to `ood_utils/saelens_adapters.py::EXPLAINER_TYPES`.
  - This is required because evaluation code branches on `isinstance(explainer, EXPLAINER_TYPES)`.

- **4) Evaluate with existing code**:
  - Pass the SAEBoost adapter into:
    - `evaluate_causal_faithfulness(...)`
    - Geometric metrics helper: RAER, hidden-space alignment
  - Store results under `results['saeboost']`.

This design keeps the **evaluation pipeline unchanged** and localizes SAEBoost-specific logic to:

- The residual training code (activations + residual targets)
- The two adapter classes that wrap base + residual explainers.

---

## 7. Summary

- **SAEBoost baseline** in TC:
  - Uses **ID-trained base explainer** (SAE or MLP-out Transcoder)
  - Adds a **small residual model** (residual SAE or residual Transcoder) trained on OOD residuals
  - Combines them via **additive reconstruction** and **feature concatenation**
- For **Transcoder**, the residual model is explicitly a **residual Transcoder**, not a SAE, to keep architecture family consistent.
- All integration is designed so that existing **metrics, hooks, and runners** can be reused with minimal code changes.

---

## 8. Concrete Implementation Plan (Files, Classes, Skeletons)

This section lists **where** to put SAEBoost code in the TC repo and provides **skeletons** for the key classes and functions. The goal is to minimize changes and keep everything localized.

### 8.1 New module: `saeboost.py`

- **File**: `saeboost.py` (at repo root, alongside `run_experiment.py`, `gae.py`, etc.)
- **Responsibility**:
  - Define:
    - `SAEBoostExplainerAdapter`
    - `SAEBoostTranscoderAdapter`
  - Optionally define:
    - Factory helpers to construct these adapters from base + residual checkpoints.

#### 8.1.1 `SAEBoostExplainerAdapter` skeleton

```python
import torch
from typing import Tuple


class SAEBoostExplainerAdapter(torch.nn.Module):
    """
    Wraps a base SAE explainer and a residual SAE explainer into a single explainer.

    Exposes the same interface as existing SAELens-based SAE adapters:
    - __call__(h, return_z=True) -> (h_recon, z)
    - decoder(z) -> h_recon
    - W_dec (dictionary), input_hook_point, output_hook_point
    """

    def __init__(self, base_explainer, resid_explainer):
        super().__init__()
        self.base_explainer = base_explainer
        self.resid_explainer = resid_explainer

        # Inherit hook metadata from base explainer
        self.input_hook_point = getattr(base_explainer, "input_hook_point", None)
        self.output_hook_point = getattr(base_explainer, "output_hook_point", None)

        # Precompute combined dictionary if available
        self._init_combined_dictionary()

    def _init_combined_dictionary(self) -> None:
        B_base = None
        B_resid = None

        # Base dictionary
        if hasattr(self.base_explainer, "saelens") and hasattr(self.base_explainer.saelens, "W_dec"):
            B_base = self.base_explainer.saelens.W_dec.T  # [d, k_base]
        elif hasattr(self.base_explainer, "W_dec"):
            B_base = self.base_explainer.W_dec.T

        # Residual dictionary
        if hasattr(self.resid_explainer, "saelens") and hasattr(self.resid_explainer.saelens, "W_dec"):
            B_resid = self.resid_explainer.saelens.W_dec.T  # [d, k_resid]
        elif hasattr(self.resid_explainer, "W_dec"):
            B_resid = self.resid_explainer.W_dec.T

        if B_base is not None and B_resid is not None:
            B_boost = torch.cat([B_base, B_resid], dim=1)  # [d, k_base + k_resid]
            # Expose as W_dec so existing geometric metrics can use it
            self.W_dec = B_boost.T  # [k_total, d]

    def forward(self, h: torch.Tensor, return_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Base SAE
        h_base, z_base = self.base_explainer(h, return_z=True)
        # Residual SAE
        h_resid, z_resid = self.resid_explainer(h, return_z=True)

        # Combine
        h_boost = h_base + h_resid
        z_boost = torch.cat([z_base, z_resid], dim=-1)

        if return_z:
            return h_boost, z_boost
        return h_boost

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        # Split features into base and residual parts
        k_base = self.base_explainer.saelens.W_dec.shape[0]
        z_base = z[..., :k_base]
        z_resid = z[..., k_base:]

        h_base = self.base_explainer.decoder(z_base)
        h_resid = self.resid_explainer.decoder(z_resid)
        return h_base + h_resid
```

> Exact attribute names (`saelens.W_dec`, `W_dec`, etc.) should be aligned with the current SAELens adapters in `ood_utils/saelens_adapters.py`.

#### 8.1.2 `SAEBoostTranscoderAdapter` skeleton

```python
import torch
from typing import Tuple


class SAEBoostTranscoderAdapter(torch.nn.Module):
    """
    Wraps a base MLP-out Transcoder and a residual Transcoder into a single explainer.

    Interface matches existing SAELens Transcoder adapters:
    - __call__(x_in, return_z=True) -> (y_recon, z)
    - decoder(z) -> y_recon
    - W_dec (dictionary), input_hook_point, output_hook_point, output_kind="mlp_out"
    """

    def __init__(self, base_transcoder, resid_transcoder):
        super().__init__()
        self.base_transcoder = base_transcoder
        self.resid_transcoder = resid_transcoder

        self.input_hook_point = getattr(base_transcoder, "input_hook_point", None)
        self.output_hook_point = getattr(base_transcoder, "output_hook_point", None)
        self.output_kind = "mlp_out"

        self._init_combined_dictionary()

    def _init_combined_dictionary(self) -> None:
        B_base = None
        B_resid = None

        if hasattr(self.base_transcoder, "W_dec"):
            B_base = self.base_transcoder.W_dec.T  # [d, k_base]
        if hasattr(self.resid_transcoder, "W_dec"):
            B_resid = self.resid_transcoder.W_dec.T  # [d, k_resid]

        if B_base is not None and B_resid is not None:
            B_boost = torch.cat([B_base, B_resid], dim=1)
            self.W_dec = B_boost.T  # [k_total, d]

    def forward(self, x_in: torch.Tensor, return_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Base Transcoder
        y_base, z_base = self.base_transcoder(x_in, return_z=True)
        # Residual Transcoder (predicts output-space residual from input_hook activations)
        y_resid, z_resid = self.resid_transcoder(x_in, return_z=True)

        y_boost = y_base + y_resid
        z_boost = torch.cat([z_base, z_resid], dim=-1)

        if return_z:
            return y_boost, z_boost
        return y_boost

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        # Split into base / residual features
        k_base = self.base_transcoder.W_dec.shape[0]
        z_base = z[..., :k_base]
        z_resid = z[..., k_base:]

        y_base = self.base_transcoder.decoder(z_base)
        y_resid = self.resid_transcoder.decoder(z_resid)
        return y_base + y_resid
```

### 8.2 Residual Training Helpers (optional)

If you prefer to keep training logic out of `run_experiment.py`, you can add a small helper module:

- **File**: `saeboost_train.py`
- **Responsibility**:
  - Functions to:
    - Collect OOD activations:
      - SAE: \(h\) at SAE hook
      - Transcoder: \((x_{\text{in}}, y_{\text{out}})\) at input/output hooks
    - Run the base explainer/transcoder to get base reconstruction
    - Construct residual targets in the corresponding output space
    - Train:
      - A residual SAE (for `explainer='sae'`)
      - A residual Transcoder (for `explainer='transcoder'`)
  - Save checkpoints following the conventions in section 5.

Skeleton outline:

```python
def train_residual_sae_for_ood(config, base_explainer, activations_loader):
    """
    Train a residual SAE on OOD residuals r = h - h_base.

    Args:
        config: hyperparameters (dict_size_resid, top_k_resid, lr, steps, etc.)
        base_explainer: frozen ID SAE explainer
        activations_loader: yields hidden activations h from OOD train split
    Returns:
        resid_explainer: trained residual SAE model or adapter
    """
    ...


def train_residual_transcoder_for_ood(config, base_transcoder, activations_loader):
    """
    Train a residual Transcoder on OOD residuals
    r = y_out - y_base, where y_base = base_transcoder(x_in).
    """
    ...
```

You can mirror the training patterns used in existing SAE / Transcoder training code, but swap the target from `h` to `r`.

### 8.3 Runner Integration (`run_experiment.py`)

In `run_experiment.py`, only minimal changes are needed:

- **Imports**:

```python
from saeboost import SAEBoostExplainerAdapter, SAEBoostTranscoderAdapter
```

- **Baseline dispatch**:
  - When parsing `--baselines`, allow `saeboost`.
  - Add a helper function:

```python
def evaluate_baseline_saeboost(
    model,
    explainer_type,  # 'sae' or 'transcoder'
    tokens_list,
    layer,
    model_type,
    device=None,
    target_mode='argmax',
    batch_id=None,
    target_token_ids=None,
    faith_m_values=None,
    faith_random_repeats=5,
    faith_m_star=32,
):
    # 1) Load base explainer (ID-trained)
    # 2) Load residual model for this (task, model, explainer, ood_set)
    # 3) Wrap with appropriate SAEBoost adapter
    # 4) Call evaluate_causal_faithfulness and geometric metrics
    ...
```

This keeps all SAEBoost-specific logic concentrated in `saeboost.py` and a small part of `run_experiment.py`, while the rest of the pipeline (data, diagnostics, metrics) remains unchanged.

## Geometry-Adaptive Explainer (GAE): Implementation Guide (Python)

This note describes how to implement the proposed **Geometry-Adaptive Explainer (GAE)** in Python.  
GAE is a **post-hoc geometric transformation** that adapts an explainer dictionary `B` to the **OOD-active hidden-space geometry** estimated from OOD hidden activations `{h_i}`. It requires **no in-distribution (ID) data** and **no retraining**.

---

### 1. Inputs / Outputs

#### Inputs
- `B`: explainer dictionary (decoder / direction matrix), shape **[d, k]**
  - columns are directions used by the explainer (e.g., SAE decoder directions, Transcoder directions)
- `H`: OOD hidden activations, shape **[N, d]**
  - each row is a hidden activation `h_i` collected from OOD data
- `r`: target subspace rank (integer), `1 <= r <= min(d, k)`
  - this is the **existing fixed-r mode** and is still fully supported
- `rank_mode` (optional): `'fixed'` (default) or `'energy'`
  - `'fixed'`: use the provided `r`
  - `'energy'`: estimate `r` from singular-value energy of \(\Delta C\)
- `rank_energy` (optional): cumulative energy target for `'energy'` mode (default `0.99`)
- `rank_delta_mode` (optional): `'ood'` or `'contrastive'`
  - `'ood'`: \(\Delta C = \widehat{\mathrm{Cov}}_{\mathrm{ood}}\)
  - `'contrastive'`: \(\Delta C = \widehat{\mathrm{Cov}}_{\mathrm{ood}} - \widehat{\mathrm{Cov}}_{\mathrm{id}}\)

#### Output
- `B_gae`: geometry-adapted dictionary, shape **[d, k]**

---

### 2. Mathematical Recipe (What to implement)

GAE computes:
1) **Reference subspace** from the explainer dictionary  
\[
U_{\text{ref}} := \mathrm{orth}(\mathrm{span}(B)) \in \mathbb{R}^{d \times r}
\]
2) **OOD-active subspace** from OOD covariance  
\[
U_{\text{ood}} := \text{Top-}r \text{ eigvecs}\big(\widehat{\mathrm{Cov}}_{\text{ood}}(h)\big) \in \mathbb{R}^{d \times r}
\]
3) **Orthogonal Procrustes alignment** between subspaces  
\[
M := U_{\text{ood}}^\top U_{\text{ref}} \in \mathbb{R}^{r \times r},\quad
M = Q \Sigma R^\top,\quad
R_{\text{align}} := Q R^\top
\]
4) **Geometric adaptation** of the dictionary  
\[
B_{\text{gae}} := U_{\text{ood}} R_{\text{align}} U_{\text{ref}}^\top B
\]

This preserves the internal coordinates of the original dictionary while realigning its geometry.

---

### 3. Practical Implementation Details

#### 3.1 Computing `U_ref = orth(span(B))`
You need an orthonormal basis for the subspace spanned by columns of `B`, truncated to rank `r`.

**Option A (SVD; recommended):**
- Compute `U, S, Vt = svd(B, full_matrices=False)`
- Set `U_ref = U[:, :r]`

This is stable and handles rank-deficient `B`.

#### 3.2 Computing OOD covariance and `U_ood`
Given `H` with shape `[N, d]`:
- Center: `Hc = H - H.mean(axis=0)`
- Covariance: `C = (Hc.T @ Hc) / (N - 1)`  (shape `[d, d]`)
- Top eigenvectors of `C`:
  - Use `eigh(C)` (symmetric) and take largest `r` eigenvectors
  - Or use randomized methods if `d` is large

**Important:** `C` is symmetric PSD; use symmetric eigensolvers (`numpy.linalg.eigh`, `torch.linalg.eigh`).

#### 3.3 Procrustes alignment (`R_align`)
Compute:
- `M = U_ood.T @ U_ref`  (shape `[r, r]`)
- `Q, _, Rt = svd(M)`
- `R_align = Q @ Rt`  (orthogonal)

This is the canonical orthogonal Procrustes solution.

#### 3.4 Final update (`B_gae`)
Compute:
- `B_gae = U_ood @ R_align @ (U_ref.T @ B)`
- Equivalent to `B_gae = (U_ood @ R_align @ U_ref.T) @ B`

---

#### 3.4.1 Optional: Theoretically-Grounded Rank Selection (`rank_mode='energy'`)

Besides fixed `r`, you can estimate rank from the spectrum of a contrast matrix \(\Delta C\):
\[
\Delta C_{(z_1)} = U_{(z_1)} \Sigma_{(z_1)} V_{(z_1)}^\top
\]

Define \(r_1\) as the smallest rank that captures 99% of squared Frobenius energy:
\[
r_1 = \min \left\{m : \frac{\sum_{i=1}^{m}\sigma_i^2}
{\sum_{i=1}^{p}\sigma_i^2} \ge 0.99 \right\},
\]
where \(\sigma_i\) are singular values of \(\Delta C_{(z_1)}\), and \(p\) is the number of non-zero singular values.

Interpretation:
- \(\sum_i \sigma_i^2 = \|\Delta C_{(z_1)}\|_F^2\) is total matrix energy.
- Top-`r` components preserving 99% energy define the effective latent subspace dimension.

Practical choices for \(\Delta C\):
- OOD-only (`rank_delta_mode='ood'`): \(\Delta C = \widehat{\mathrm{Cov}}_{\mathrm{ood}}\)
- Contrastive (`rank_delta_mode='contrastive'`): \(\Delta C = \widehat{\mathrm{Cov}}_{\mathrm{ood}} - \widehat{\mathrm{Cov}}_{\mathrm{id}}\)

For symmetric \(\Delta C\), you may use eigendecomposition (`eigvalsh`) instead of full SVD.
Singular values are \(|\lambda_i|\), so energy is \(\lambda_i^2\).

Recommended guardrails:
- `r <= rank(B)` (numerical matrix rank of dictionary),
- `r <= N_ood - 1` (sample covariance rank limit),
- optional bounds `rank_min`, `rank_max`.

This is now available in `gae.py` while preserving the original fixed-r behavior.

---

### 3.5 Decoder-Only Refit on OOD (Closed-Form Calibration)

In some OOD settings, directly applying the geometry-adapted dictionary `B_gae`
may lead to poor hidden-state reconstruction, resulting in large reconstruction
error and unstable faithfulness evaluation.
To address this issue **without retraining the explainer**, we apply a
**decoder-only refit** on OOD activations.

This procedure calibrates the decoding operator while keeping the
feature extraction and geometry adaptation fixed.

---

#### Motivation

GAE aligns the *geometry* of the explainer dictionary to the OOD-active subspace.
However, the original decoder may become poorly conditioned under this alignment,
leading to:
- large hidden reconstruction error (`ReconErr ≫ 1`),
- significant logit drift even without ablation (`LogitErr0 > 1`).

In such cases, ablation-based faithfulness metrics become ill-defined.
Decoder-only refit restores a stable reconstruction baseline on OOD data.

---

#### Formulation

Let:
- \( h_i \in \mathbb{R}^d \): OOD hidden activations,
- \( a_i \in \mathbb{R}^k \): explainer features extracted using the (fixed) GAE dictionary,
  e.g. \( a_i = \phi(h_i; B_{\mathrm{gae}}) \).

We solve the following **least-squares problem**:
\[
D^\star
= \arg\min_D \sum_i \| h_i - D a_i \|_2^2,
\]
which has the closed-form solution:
\[
D^\star = H^\top A (A^\top A + \lambda I)^{-1},
\]
where:
- \( H \in \mathbb{R}^{N \times d} \) stacks OOD hidden activations,
- \( A \in \mathbb{R}^{N \times k} \) stacks corresponding explainer features,
- \( \lambda \ge 0 \) is a small ridge term for numerical stability.

Only the decoder \(D\) is updated; the explainer features and GAE alignment remain fixed.

---

#### PyTorch Implementation

```python
@torch.no_grad()
def decoder_only_refit(
    H: torch.Tensor,   # [N, d] OOD hidden activations
    A: torch.Tensor,   # [N, k] explainer features from B_gae
    lam: float = 1e-4  # ridge regularization
) -> torch.Tensor:
    """
    Closed-form decoder calibration on OOD activations.

    Returns:
        D_star: [d, k] calibrated decoder
    """
    # A^T A + λI
    AtA = A.T @ A
    reg = lam * torch.eye(AtA.shape[0], device=A.device)
    inv = torch.linalg.inv(AtA + reg)

    # H^T A (A^T A + λI)^{-1}
    D_star = H.T @ A @ inv
    return D_star
````

---

#### Usage in Evaluation

1. Compute `B_gae` using GAE.
2. Extract OOD features:

   * `A = features(h; B_gae)`
3. Compute calibrated decoder `D_star` using the above routine.
4. During faithfulness evaluation:

   * use **`B_gae` for feature extraction and scoring**,
   * use **`D_star` for reconstruction after ablation**.

This ensures:

* reconstruction is stable on OOD,
* ablation effects are measured relative to a valid baseline,
* no explainer retraining is performed.

---

#### Implementation Status (Applied)

The decoder-only refit path is now wired into the experiment pipeline:

1. `run_experiment.py::evaluate_baseline_gae` now computes `D_star` with
   `decoder_only_refit(H_ood, A_ood, lam)` after `B_gae` is built.
2. The adapted explainer stores:
   - `explainer.B_gae` (feature extraction dictionary),
   - `explainer.D_gae` (reconstruction decoder),
   - `explainer.use_gae = True`.
3. Faithfulness/diagnostics code automatically uses `D_gae` when present,
   so interventions use a matched `(B_gae, D_gae)` pair instead of mixing
   `B_gae` with the original decoder.

Fair-evaluation note:
- `D_star` is fitted only from OOD **train/adaptation activations** and does
  not use evaluation/test prompts or labels.
- This keeps GAE in the same adaptation regime (test-time-free, unlabeled)
  while reducing reconstruction-mismatch artifacts in nAOPC.

---

#### Interpretation and Scope

* Decoder-only refit is a **post-hoc calibration**, not a new explainer.
* It uses no ID data and requires no gradient-based optimization.
* The procedure preserves the core claim of GAE:
  *geometry adaptation without retraining*.

In experiments, this step significantly reduces reconstruction error and restores
meaningful ablation-based faithfulness evaluation in challenging OOD regimes.


---

## 4. PyTorch Version (GPU-friendly)

```python
import torch

@torch.no_grad()
def gae_torch(B: torch.Tensor, H: torch.Tensor, r: int) -> torch.Tensor:
    """
    GAE in PyTorch.

    Args:
        B: [d, k]
        H: [N, d]
        r: target rank
    Returns:
        B_gae: [d, k]
    """
    d, k = B.shape
    N, d2 = H.shape
    assert d == d2
    assert 1 <= r <= min(d, k)

    # 1) U_ref via SVD
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U_ref = U[:, :r]  # [d, r]

    # 2) Cov + U_ood
    Hc = H - H.mean(dim=0, keepdim=True)
    denom = max(N - 1, 1)
    C = (Hc.T @ Hc) / denom  # [d, d]
    evals, evecs = torch.linalg.eigh(C)  # ascending
    idx = torch.argsort(evals, descending=True)
    U_ood = evecs[:, idx[:r]]  # [d, r]

    # 3) Procrustes
    M = U_ood.T @ U_ref        # [r, r]
    Q, _, Vh2 = torch.linalg.svd(M, full_matrices=False)
    R_align = Q @ Vh2          # [r, r]

    # 4) Update
    B_gae = U_ood @ R_align @ (U_ref.T @ B)  # [d, k]
    return B_gae
```

### 4.1 Optional Auto-Rank Usage (added, fixed-r still available)

`gae.py` now supports both modes:

```python
# Existing fixed-r usage (unchanged)
B_gae, B_original, D_star = apply_gae_to_explainer(
    explainer,
    id_activations,
    ood_activations,
    r=128,
    model_type="transcoder",
    rank_mode="fixed",
)

# New energy-based usage
B_gae, B_original, D_star = apply_gae_to_explainer(
    explainer,
    id_activations,
    ood_activations,
    model_type="transcoder",
    rank_mode="energy",
    rank_energy=0.99,
    rank_delta_mode="ood",  # or "contrastive"
    rank_min=1,
    rank_max=None,
)
```


## 5. Sanity Checks (Recommended)

After computing B_gae, validate:

1. Shape check

- B_gae.shape == B.shape

2. Subspace check

- span(B_gae) should be close to U_ood (up to rank r)

    - e.g., compute U_new = svd(B_gae)[0][:, :r]

    - check projector distance ||U_new U_new^T - U_ood U_ood^T||_2 decreases vs U_ref

3. Orthogonality check

- R_align.T @ R_align ≈ I

4. Numerical stability

- ensure N is reasonably large; if N is small, covariance is noisy

- you may use shrinkage covariance if needed:

    - C <- (1-λ)C + λ I

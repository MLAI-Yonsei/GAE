"""
Geometry-Adaptive Explainer (GAE) for TransformerLens models.
"""
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config

def _empirical_covariance(H: torch.Tensor) -> torch.Tensor:
    """
    Empirical covariance of hidden activations H with rows as samples.
    """
    Hc = H - H.mean(dim=0, keepdim=True)
    denom = max(H.shape[0] - 1, 1)
    return (Hc.T @ Hc) / denom


@torch.no_grad()
def estimate_rank_from_delta_cov(
    delta_cov: torch.Tensor,
    energy_ratio: float = 0.99,
    min_rank: int = 1,
    max_rank: int = None,
    eps: float = 1e-12,
):
    """
    Estimate rank by cumulative squared singular-value energy of ΔC.

    For symmetric ΔC, singular values are |eigenvalues|. We therefore use
    eigvalsh + abs(.) and accumulate λ_i^2 energy.

    Returns:
        r: estimated rank
        captured: cumulative energy ratio captured by top-r components
    """
    if not (0.0 < energy_ratio <= 1.0):
        raise ValueError(f"energy_ratio must be in (0, 1], got {energy_ratio}")

    d = delta_cov.shape[0]
    if max_rank is None:
        max_rank = d
    max_rank = max(1, min(int(max_rank), d))
    min_rank = max(1, min(int(min_rank), max_rank))

    # Symmetrize for numerical stability before eigendecomposition.
    delta_sym = 0.5 * (delta_cov + delta_cov.T)
    evals = torch.linalg.eigvalsh(delta_sym.to(torch.float32))
    sigmas = torch.sort(torch.abs(evals), descending=True).values
    energy = sigmas.pow(2)
    total = energy.sum()

    if total <= eps:
        return min_rank, 0.0

    cum = torch.cumsum(energy, dim=0) / (total + eps)
    target = torch.tensor(energy_ratio, device=cum.device, dtype=cum.dtype)
    r_energy = int(torch.searchsorted(cum, target).item()) + 1  # 1-based
    r = max(min_rank, min(r_energy, max_rank))
    captured = float(cum[r - 1].item())
    return r, captured


@torch.no_grad()
def select_rank_for_gae(
    B: torch.Tensor,
    H_ood: torch.Tensor,
    H_id: torch.Tensor = None,
    energy_ratio: float = 0.99,
    delta_mode: str = "ood",
    min_rank: int = 1,
    max_rank: int = None,
):
    """
    Auto-select GAE rank from ΔC using an energy criterion.

    delta_mode:
        - "ood":         ΔC = Cov_ood
        - "contrastive": ΔC = Cov_ood - Cov_id
    """
    d, k = B.shape
    cov_ood = _empirical_covariance(H_ood)

    if delta_mode == "ood":
        delta_cov = cov_ood
    elif delta_mode == "contrastive":
        if H_id is None:
            raise ValueError("H_id is required when delta_mode='contrastive'")
        delta_cov = cov_ood - _empirical_covariance(H_id)
    else:
        raise ValueError(f"Unknown delta_mode='{delta_mode}'. Use 'ood' or 'contrastive'.")

    # Conservative rank cap from dictionary rank and OOD sample rank.
    rank_B = int(torch.linalg.matrix_rank(B).item())
    rank_sample = max(int(H_ood.shape[0]) - 1, 1)
    rank_cap = min(d, k, rank_B, rank_sample)
    if max_rank is not None:
        rank_cap = min(rank_cap, int(max_rank))
    rank_cap = max(int(min_rank), rank_cap)

    r, captured = estimate_rank_from_delta_cov(
        delta_cov=delta_cov,
        energy_ratio=energy_ratio,
        min_rank=min_rank,
        max_rank=rank_cap,
    )

    info = {
        "delta_mode": delta_mode,
        "energy_ratio_target": float(energy_ratio),
        "energy_ratio_captured": float(captured),
        "rank_cap": int(rank_cap),
    }
    return r, info


@torch.no_grad()
def gae_torch(
    B: torch.Tensor,
    H: torch.Tensor,
    r: int,
    recon_lambda: float = 0.0,
    Z: Optional[torch.Tensor] = None,
    H_target: Optional[torch.Tensor] = None,
    b_target: Optional[torch.Tensor] = None,
):
    """
    GAE in PyTorch.

    Computes geometry-adapted dictionary B_gae from original dictionary B
    and OOD hidden activations H.

    Args:
        B: Explainer dictionary, shape [d, k]
        H: OOD hidden activations, shape [N, d]
        r: Target subspace rank (integer), 1 <= r <= min(d, k)
        recon_lambda: Weight for reconstruction term added to the Procrustes
            target M. When 0.0 (default), reduces exactly to the legacy
            feature-preservation target.
        Z: Optional [N_rec, k] encoder features from the frozen ID explainer
            evaluated on OOD activations. Required when recon_lambda > 0.
        H_target: Optional [N_rec, d] decoder-output targets (mlp_out for
            transcoder, h for SAE). Required when recon_lambda > 0.
        b_target: Optional [d] decoder-output bias (b_dec_out for transcoder,
            b_dec for SAE). Required when recon_lambda > 0.

    Returns:
        B_gae: Geometry-adapted dictionary, shape [d, k]
        gae_info: dict with intermediate rotation components
            U_ood [d, r], U_ref [d, r], R_align [r, r], plus recon stats.

    Mathematical steps:
    1) U_ref = orth(span(B)) via SVD, truncated to rank r
    2) U_ood = Top-r eigvecs(Cov_ood(h)) from OOD covariance
    3) M_pres = U_ood^T U_ref; optionally add normalized M_rec
    4) R_align = U V^T (standard orthogonal Procrustes) on combined M
    5) B_gae = U_ood @ R_align @ (U_ref.T @ B)
    """
    d, k = B.shape
    N, d2 = H.shape
    assert d == d2, f"Dimension mismatch: B has {d} rows, H has {d2} columns"
    assert 1 <= r <= min(d, k), f"Rank r={r} must satisfy 1 <= r <= min(d={d}, k={k})"

    # Ensure tensors are on the same device
    device = B.device
    H = H.to(device)

    # 1) U_ref via SVD
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U_ref = U[:, :r]  # [d, r]

    # 2) Cov + U_ood
    Hc = H - H.mean(dim=0, keepdim=True)
    denom = max(N - 1, 1)
    C = (Hc.T @ Hc) / denom  # [d, d]
    evals, evecs = torch.linalg.eigh(C)  # ascending order
    idx = torch.argsort(evals, descending=True)
    U_ood = evecs[:, idx[:r]]  # [d, r]

    # 3) Procrustes target (feature preservation; optionally + reconstruction)
    M = U_ood.T @ U_ref  # [r, r]

    recon_info = {"recon_lambda_used": float(recon_lambda)}
    if (
        recon_lambda > 0.0
        and Z is not None
        and H_target is not None
        and b_target is not None
    ):
        Z_dev = Z.to(device=device, dtype=B.dtype)
        H_tgt_dev = H_target.to(device=device, dtype=B.dtype)
        b_tgt_dev = b_target.to(device=device, dtype=B.dtype)
        N_rec = Z_dev.shape[0]
        assert N_rec == H_tgt_dev.shape[0], (
            f"N mismatch: Z={tuple(Z_dev.shape)}, H_target={tuple(H_tgt_dev.shape)}"
        )
        BT_Uref = B.T @ U_ref  # [k, r]
        S_xx = torch.zeros(r, r, device=device, dtype=B.dtype)
        S_xe = torch.zeros(r, d, device=device, dtype=B.dtype)
        CHUNK = 4096
        for start in range(0, N_rec, CHUNK):
            end = min(start + CHUNK, N_rec)
            Z_chunk = Z_dev[start:end]                  # [c, k]
            H_chunk = H_tgt_dev[start:end]              # [c, d]
            X_chunk = Z_chunk @ BT_Uref                 # [c, r]
            W_chunk = Z_chunk @ B.T                     # [c, d]
            E_chunk = H_chunk - b_tgt_dev.unsqueeze(0) - W_chunk
            S_xx += X_chunk.T @ X_chunk
            S_xe += X_chunk.T @ E_chunk
            del Z_chunk, X_chunk, W_chunk, E_chunk, H_chunk
        S_xx /= N_rec
        S_xe /= N_rec
        M_rec_raw = S_xx @ (U_ref.T @ U_ood) + S_xe @ U_ood  # [r, r]
        tau_rec = torch.trace(S_xx) / r
        M_rec = M_rec_raw / (tau_rec + 1e-8)
        M = M + recon_lambda * M_rec
        recon_info.update({
            "tau_rec": float(tau_rec.item()),
            "M_rec_frobenius": float(M_rec_raw.norm().item()),
            "M_rec_norm_frobenius": float(M_rec.norm().item()),
            "N_for_recon": int(N_rec),
        })

    # 4) Orthogonal Procrustes: R* = U V^T (standard convention).
    # torch.linalg.svd returns Vh = V^T, so Q @ Vh2 = U V^T directly.
    Q, _, Vh2 = torch.linalg.svd(M, full_matrices=False)
    R_align = Q @ Vh2  # [r, r]

    # 5) Update
    B_gae = U_ood @ R_align @ (U_ref.T @ B)  # [d, k]

    gae_info = {
        "U_ood": U_ood,      # [d, r]
        "U_ref": U_ref,      # [d, r]
        "R_align": R_align,  # [r, r]
        "procrustes_convention": "UV^T",
    }
    gae_info.update(recon_info)
    return B_gae, gae_info


@torch.no_grad()
def consistent_rotation_decoder(B_orig: torch.Tensor, gae_info: dict) -> torch.Tensor:
    """
    Apply the same subspace rotation as GAE encoder to the decoder.

    When GAE rotates B via B_gae = U_ood @ T* @ U_ref^T @ B,
    the decoder must undergo the same basis change to maintain
    encoder-decoder consistency (analogous to cross-lingual alignment).

    Args:
        B_orig: Original dictionary [d, k] (serves as ID decoder)
        gae_info: dict from gae_torch with U_ood, U_ref, R_align

    Returns:
        D_consistent: Rotated decoder [d, k]
    """
    U_ood = gae_info["U_ood"].to(B_orig.device)
    U_ref = gae_info["U_ref"].to(B_orig.device)
    R_align = gae_info["R_align"].to(B_orig.device)
    return U_ood @ R_align @ (U_ref.T @ B_orig)  # [d, k]


@torch.no_grad()
def decoder_only_refit(
    H: torch.Tensor,   # [N, d] OOD hidden activations
    A: torch.Tensor,   # [N, k] explainer features from B_gae
    lam: float = 1e-4,  # ridge regularization
    method: str = "auto",
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Closed-form decoder calibration on OOD activations.

    Args:
        H: [N, d] OOD hidden activations
        A: [N, k] explainer features extracted with B_gae
        lam: ridge regularization coefficient

    Returns:
        D_star: [d, k] calibrated decoder
    """
    if method not in ("auto", "normal", "row"):
        raise ValueError(f"Unknown decoder refit method='{method}'. Use 'auto', 'normal', or 'row'.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    H = H.to(dtype=torch.float32)
    A = A.to(device=H.device, dtype=torch.float32)
    n, k = A.shape
    d = H.shape[1]

    if method == "row":
        use_row = True
    elif method == "normal":
        use_row = False
    else:
        use_row = n < k

    if use_row:
        # Row-space solve:
        # D* = H^T (A A^T + λI)^(-1) A
        AAT = A @ A.T  # [N, N]
        # Numerical symmetrization + adaptive diagonal jitter for robust factorization.
        AAT = 0.5 * (AAT + AAT.T)
        if not torch.isfinite(AAT).all():
            raise RuntimeError("decoder_only_refit: AAT contains NaN/Inf values.")

        diag_mean = float(torch.diagonal(AAT).mean().item()) if n > 0 else 0.0
        jitter = max(float(lam), 1e-8 * max(diag_mean, 1.0))
        eye_n = torch.eye(n, device=A.device, dtype=A.dtype)
        L = None
        AAT_reg = None
        for _ in range(8):
            AAT_reg = AAT + jitter * eye_n
            L_try, info = torch.linalg.cholesky_ex(AAT_reg, check_errors=False)
            if int(info.item()) == 0:
                L = L_try
                break
            jitter *= 10.0

        Ht = H.T.contiguous()  # [d, N]
        D_star = torch.empty((d, k), device=A.device, dtype=A.dtype)
        if L is not None:
            for start in range(0, k, int(chunk_size)):
                end = min(k, start + int(chunk_size))
                A_blk = A[:, start:end]                  # [N, c]
                X_blk = torch.cholesky_solve(A_blk, L)  # [N, c]
                D_star[:, start:end] = Ht @ X_blk       # [d, c]
            return D_star

        # Fallback when Cholesky remains unstable: robust linear solve with final jitter.
        for start in range(0, k, int(chunk_size)):
            end = min(k, start + int(chunk_size))
            A_blk = A[:, start:end]  # [N, c]
            try:
                X_blk = torch.linalg.solve(AAT_reg, A_blk)
            except RuntimeError:
                X_blk = torch.linalg.lstsq(AAT_reg, A_blk).solution
            D_star[:, start:end] = Ht @ X_blk
        return D_star

    # Normal-equation solve:
    # D* = H^T A (A^T A + λI)^(-1)
    ATA = A.T @ A  # [k, k]
    if lam > 0:
        ATA = ATA + lam * torch.eye(k, device=A.device, dtype=A.dtype)
    RHS = A.T @ H  # [k, d]
    X = torch.linalg.solve(ATA, RHS)  # [k, d]
    D_star = X.T  # [d, k]
    return D_star


@torch.no_grad()
def adapt_features_for_decoder(
    z0: torch.Tensor,
    h: torch.Tensor,
    decoder: torch.Tensor,
    adapter_mode: Optional[str] = None,
    gains: Optional[torch.Tensor] = None,
    ridge: float = 1e-3,
    max_support: int = 128,
) -> torch.Tensor:
    """
    Adapt encoder features for a decoder-space GAE variant.

    Modes:
        None: return z0 unchanged
        'diag': apply global nonnegative feature gains
        'support_refit': preserve sparse support, but refit coefficients under decoder
    """
    if adapter_mode in (None, "", "none"):
        return z0

    z0 = z0.to(decoder.device)
    h = h.to(decoder.device)

    if adapter_mode == "diag":
        if gains is None:
            raise ValueError("gains are required for adapter_mode='diag'")
        gains = gains.to(z0.device, dtype=z0.dtype)
        return torch.relu(z0 * gains.unsqueeze(0))

    if adapter_mode != "support_refit":
        raise ValueError(f"Unknown adapter_mode='{adapter_mode}'")

    batch, k = z0.shape
    z_out = torch.zeros_like(z0)
    max_support = max(1, min(int(max_support), k))
    eye_cache = {}

    for b in range(batch):
        z_b = z0[b]
        support = torch.nonzero(z_b > 0, as_tuple=False).flatten()
        if support.numel() == 0:
            continue
        if support.numel() > max_support:
            vals = z_b[support]
            top_local = torch.topk(vals, k=max_support, largest=True).indices
            support = support[top_local]

        D_s = decoder[:, support]  # [d, s]
        z_s0 = z_b[support]        # [s]
        s = int(support.numel())
        if s not in eye_cache:
            eye_cache[s] = torch.eye(s, device=decoder.device, dtype=decoder.dtype)
        lhs = D_s.T @ D_s + float(ridge) * eye_cache[s]
        rhs = D_s.T @ h[b].to(decoder.dtype) + float(ridge) * z_s0.to(decoder.dtype)
        try:
            z_s = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            z_s = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)
        z_out[b, support] = torch.relu(z_s.to(z_out.dtype))

    return z_out


@torch.no_grad()
def fit_diag_feature_gains(
    explainer,
    activations_in: torch.Tensor,
    activations_target: torch.Tensor,
    decoder: torch.Tensor,
    lam: float = 1e-3,
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Fit nonnegative per-feature gains s for h ~= (s ⊙ z_id) decoder^T.

    Uses a diagonal approximation to the full least-squares problem:
        s_j = <h, z_j d_j> / (||d_j||^2 ||z_j||^2 + lam)
    """
    device = decoder.device
    decoder = decoder.to(device=device, dtype=torch.float32)
    activations_in = activations_in.to(device=device, dtype=torch.float32)
    activations_target = activations_target.to(device=device, dtype=torch.float32)
    k = decoder.shape[1]
    d_norm_sq = decoder.pow(2).sum(dim=0)  # [k]

    numer = torch.zeros(k, device=device, dtype=torch.float32)
    denom = torch.full((k,), float(lam), device=device, dtype=torch.float32)

    for start in range(0, int(activations_in.shape[0]), int(batch_size)):
        end = min(int(activations_in.shape[0]), start + int(batch_size))
        h_in = activations_in[start:end]
        h_tgt = activations_target[start:end]
        _, z_b = explainer(h_in, return_z=True)
        z_b = z_b.to(device=device, dtype=torch.float32)
        h_proj = h_tgt @ decoder  # [B, k], each column is h·d_j
        numer += (z_b * h_proj).sum(dim=0)
        denom += d_norm_sq * z_b.pow(2).sum(dim=0)

    gains = numer / denom.clamp_min(1e-8)
    return gains.clamp_min(0.0)


@torch.no_grad()
def select_decoder_mix_alpha(
    explainer,
    activations_in: torch.Tensor,
    activations_target: torch.Tensor,
    B_original: torch.Tensor,
    B_gae: torch.Tensor,
    decoder_bias: Optional[torch.Tensor] = None,
    alpha_grid: Optional[Iterable[float]] = None,
    batch_size: int = 128,
) -> tuple[float, torch.Tensor]:
    """
    Choose a scalar interpolation alpha between the original and GAE decoders
    using OOD reconstruction error under the original encoder features.
    """
    if alpha_grid is None:
        alpha_grid = [i / 10.0 for i in range(11)]

    device = B_original.device
    activations_in = activations_in.to(device=device, dtype=torch.float32)
    activations_target = activations_target.to(device=device, dtype=torch.float32)
    B_original = B_original.to(device=device, dtype=torch.float32)
    B_gae = B_gae.to(device=device, dtype=torch.float32)
    if decoder_bias is not None:
        decoder_bias = decoder_bias.to(device=device, dtype=torch.float32)

    best_alpha = 0.0
    best_err = float("inf")
    best_decoder = B_original

    for alpha in alpha_grid:
        alpha = float(alpha)
        D_mix = (1.0 - alpha) * B_original + alpha * B_gae
        err_sum = 0.0
        n_seen = 0
        for start in range(0, int(activations_in.shape[0]), int(batch_size)):
            end = min(int(activations_in.shape[0]), start + int(batch_size))
            h_in = activations_in[start:end]
            h_tgt = activations_target[start:end]
            _, z_b = explainer(h_in, return_z=True)
            z_b = z_b.to(device=device, dtype=torch.float32)
            h_hat = z_b @ D_mix.T
            if decoder_bias is not None:
                h_hat = h_hat + decoder_bias
            err_sum += float(torch.sum((h_tgt - h_hat).pow(2)).item())
            n_seen += int(h_tgt.shape[0])
        mse = err_sum / max(n_seen, 1)
        if mse < best_err:
            best_err = mse
            best_alpha = alpha
            best_decoder = D_mix.clone()

    return best_alpha, best_decoder


@torch.no_grad()
def fit_affine_constrained_decoder(
    explainer,
    activations_in: torch.Tensor,
    activations_target: torch.Tensor,
    B_original: torch.Tensor,
    B_gae: torch.Tensor,
    U_ood: torch.Tensor,
    lam_geom: float = 1e-1,
    lam_id: float = 1e-2,
    lam_gae: float = 1e-1,
    fit_samples: Optional[int] = 4096,
    batch_size: int = 128,
    solver_device: str = "cpu",
    match_column_norms: bool = False,
    decoder_mix_prior: float = 0.0,
    ce_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Fit an affine decoder h ~= z D^T + b under GAE geometry constraints.

    Objective:
        min_{D,b} (1/N) sum_j w_j ||[H_c]_j - Z_c (D^T)_j||^2
                  + lam_geom ||(I - P_ood) D||_F^2
                  + lam_id ||D - B_original||_F^2
                  + lam_gae ||D - B_gae||_F^2

    where P_ood = U_ood U_ood^T and w_j are per-dimension CE sensitivity
    weights (default: uniform w_j = 1). When ce_weights is provided, the
    solver uses per-row Q-basis weights w_tilde_m = sum_j Q_{jm}^2 * w_j
    (diagonal approximation of Q^T diag(w) Q).

    The solve is carried out in the OOD basis so the geometry penalty reduces
    to two ridge levels: one inside span(U_ood) and one outside.
    """
    if fit_samples is not None and int(fit_samples) > 0:
        n_fit = min(int(fit_samples), int(activations_in.shape[0]), int(activations_target.shape[0]))
        activations_in = activations_in[:n_fit]
        activations_target = activations_target[:n_fit]

    model_device = B_original.device
    solver_device = torch.device(solver_device)
    work_dtype = torch.float32

    activations_in = activations_in.to(device=model_device, dtype=work_dtype)
    activations_target = activations_target.to(device=solver_device, dtype=work_dtype)
    B_original = B_original.to(device=solver_device, dtype=work_dtype)
    B_gae = B_gae.to(device=solver_device, dtype=work_dtype)
    U_ood = U_ood.to(device=solver_device, dtype=work_dtype)

    z_batches = []
    for start in range(0, int(activations_in.shape[0]), int(batch_size)):
        end = min(int(activations_in.shape[0]), start + int(batch_size))
        _, z_b = explainer(activations_in[start:end], return_z=True)
        z_batches.append(z_b.detach().to(device=solver_device, dtype=work_dtype))
    Z = torch.cat(z_batches, dim=0)  # [N, k]
    H = activations_target  # [N, d]

    mu_z = Z.mean(dim=0)
    mu_h = H.mean(dim=0)
    Zc = Z - mu_z
    Hc = H - mu_h

    lam_id = float(lam_id)
    lam_gae = float(lam_gae)
    lam_geom = float(lam_geom)

    # === SAE GAE Module 1B: degenerate solver toggles (env-controlled) ===
    # All defaults are OFF — preserves current TC behavior. Used for staged
    # ablation in docs/sae_diag/phase_reports/02_phase1_conclusion_and_phase2_plan.md
    _degen_no_qbasis = os.environ.get("GAE_AC_DEGEN_NO_QBASIS", "0") == "1"
    _degen_no_center = os.environ.get("GAE_AC_DEGEN_NO_CENTER", "0") == "1"
    _degen_wdec_only = os.environ.get("GAE_AC_DEGEN_PRIOR_WDEC_ONLY", "0") == "1"
    _degen_no_geom = os.environ.get("GAE_AC_DEGEN_NO_GEOM", "0") == "1"
    _degen_no_mixprior = os.environ.get("GAE_AC_DEGEN_NO_MIXPRIOR", "0") == "1"

    if _degen_wdec_only:
        lam_gae = 0.0
    if _degen_no_geom:
        lam_geom = 0.0
    if _degen_no_mixprior:
        decoder_mix_prior = 0.0
    if _degen_no_center:
        mu_z = torch.zeros_like(mu_z)
        mu_h = torch.zeros_like(mu_h)
        Zc = Z.clone()
        Hc = H.clone()

    # Weighted decoder prior from ID and GAE dictionaries.
    lam_prior = lam_id + lam_gae
    if lam_prior > 0:
        P0 = (lam_id * B_original + lam_gae * B_gae) / lam_prior
    else:
        P0 = torch.zeros_like(B_original)

    _degen_info = {
        "degen_mode_active": any([_degen_no_qbasis, _degen_no_center, _degen_wdec_only,
                                  _degen_no_geom, _degen_no_mixprior]),
        "degen_no_qbasis": _degen_no_qbasis,
        "degen_no_center": _degen_no_center,
        "degen_wdec_only": _degen_wdec_only,
        "degen_no_geom": _degen_no_geom,
        "degen_no_mixprior": _degen_no_mixprior,
    }
    if _degen_info["degen_mode_active"]:
        print(f"[GAE] Degenerate solver toggles active (lam_id={lam_id}, lam_gae={lam_gae}, "
              f"lam_geom={lam_geom})")

    # === Simple OLS branch (Q-basis bypass) ===
    if _degen_no_qbasis:
        k = Z.shape[1]
        ridge_eff = max(float(lam_prior), 1e-6)
        ZtZ = Zc.T @ Zc + ridge_eff * torch.eye(k, device=solver_device, dtype=work_dtype)
        ZtH_with_prior = Zc.T @ Hc + ridge_eff * P0.T  # [k, d]
        D_T = torch.linalg.solve(ZtZ, ZtH_with_prior)  # [k, d]
        D_star = D_T.T  # [d, k]

        decoder_mix_prior_eff = float(max(0.0, min(1.0, decoder_mix_prior)))
        if decoder_mix_prior_eff > 0.0 and lam_prior > 0.0:
            D_star = (1.0 - decoder_mix_prior_eff) * D_star + decoder_mix_prior_eff * P0
        if bool(match_column_norms):
            target_norms = B_original.norm(dim=0).clamp_min(1e-8)
            current_norms = D_star.norm(dim=0).clamp_min(1e-8)
            D_star = D_star * (target_norms / current_norms).unsqueeze(0)

        b_star = mu_h - (D_star @ mu_z)

        H_hat = Z @ D_star.T + b_star.unsqueeze(0)
        mse = float(torch.mean((H_hat - H).pow(2)).item())
        info = {
            "fit_samples": int(Z.shape[0]),
            "rank": int(U_ood.shape[1]),
            "lam_geom": lam_geom,
            "lam_id": lam_id,
            "lam_gae": lam_gae,
            "fit_mse": mse,
            "match_column_norms": bool(match_column_norms),
            "decoder_mix_prior": decoder_mix_prior_eff,
            "solver_path": "simple_ols_degenerate",
            "ce_weights_enabled": ce_weights is not None,
        }
        info.update(_degen_info)
        return D_star, b_star, info
    # === End simple OLS branch ===

    # Complete orthonormal basis whose leading columns span U_ood.
    Q, _ = torch.linalg.qr(U_ood, mode="complete")  # [d, d]
    r = int(U_ood.shape[1])

    Hq = Hc @ Q               # [N, d]
    Pq = Q.T @ P0             # [d, k]
    d = Q.shape[0]
    N = Zc.shape[0]

    # CE-sensitivity weighting in Q-basis
    ce_weight_info = {"ce_weights_enabled": ce_weights is not None}
    if ce_weights is not None:
        w = ce_weights.to(device=solver_device, dtype=work_dtype)
        w_tilde = (Q.T * Q.T) @ w  # [d] — diagonal of Q^T diag(w) Q
        ce_weight_info.update({
            "w_tilde_min": float(w_tilde.min().item()),
            "w_tilde_max": float(w_tilde.max().item()),
            "w_tilde_mean": float(w_tilde.mean().item()),
        })
    else:
        w_tilde = None

    def _solve_group(target_q: torch.Tensor, prior_q: torch.Tensor,
                     ridge: float, row_weights: Optional[torch.Tensor] = None):
        m = target_q.shape[1] if target_q.dim() == 2 else 0
        if m == 0:
            return target_q.new_zeros((0, prior_q.shape[1]))
        ridge_eff = max(float(ridge), 1e-6)

        if row_weights is None:
            # Original batch solve (unchanged)
            residual = target_q - (Zc @ prior_q.T)  # [N, m]
            ZZt = Zc @ Zc.T
            ZZt = 0.5 * (ZZt + ZZt.T)
            ZZt = ZZt + ridge_eff * torch.eye(ZZt.shape[0], device=ZZt.device, dtype=ZZt.dtype)
            alpha = torch.linalg.solve(ZZt, residual)  # [N, m]
            return prior_q + alpha.T @ Zc  # [m, k]

        # Per-row weighted solve via SVD of Zc
        k = prior_q.shape[1]
        U_z, S_z, Vh_z = torch.linalg.svd(Zc, full_matrices=False)
        Lambda = S_z.pow(2)
        ZtH = Zc.T @ target_q
        ZtH_v = Vh_z @ ZtH
        P_v = Vh_z @ prior_q.T
        w_vec = row_weights.unsqueeze(0)
        diag_all = w_vec * Lambda.unsqueeze(1) + ridge_eff
        rhs_all = w_vec * ZtH_v + lam_prior * P_v
        result_v = rhs_all / diag_all
        Dq_group = (Vh_z.T @ result_v).T
        rank_svd = min(N, k)
        if rank_svd < k:
            for col in range(m):
                p_col = prior_q[col, :]
                p_in_range = Vh_z.T @ (Vh_z @ p_col)
                p_null = p_col - p_in_range
                Dq_group[col, :] += (lam_prior / ridge_eff) * p_null
        return Dq_group

    if w_tilde is not None:
        Dq_in = _solve_group(Hq[:, :r], Pq[:r, :], lam_prior, row_weights=w_tilde[:r])
        Dq_out = _solve_group(Hq[:, r:], Pq[r:, :], lam_prior + lam_geom, row_weights=w_tilde[r:])
    else:
        Dq_in = _solve_group(Hq[:, :r], Pq[:r, :], lam_prior)
        Dq_out = _solve_group(Hq[:, r:], Pq[r:, :], lam_prior + lam_geom)
    Dq = torch.cat([Dq_in, Dq_out], dim=0)
    D_star = Q @ Dq

    decoder_mix_prior = float(max(0.0, min(1.0, decoder_mix_prior)))
    if decoder_mix_prior > 0.0 and lam_prior > 0.0:
        D_star = (1.0 - decoder_mix_prior) * D_star + decoder_mix_prior * P0

    if bool(match_column_norms):
        target_norms = B_original.norm(dim=0).clamp_min(1e-8)
        current_norms = D_star.norm(dim=0).clamp_min(1e-8)
        D_star = D_star * (target_norms / current_norms).unsqueeze(0)

    b_star = mu_h - (D_star @ mu_z)

    H_hat = Z @ D_star.T + b_star.unsqueeze(0)
    mse = float(torch.mean((H_hat - H).pow(2)).item())
    info = {
        "fit_samples": int(Z.shape[0]),
        "rank": int(r),
        "lam_geom": lam_geom,
        "lam_id": lam_id,
        "lam_gae": lam_gae,
        "fit_mse": mse,
        "match_column_norms": bool(match_column_norms),
        "decoder_mix_prior": decoder_mix_prior,
        "solver_path": "qbasis_default",
    }
    info.update(ce_weight_info)
    info.update(_degen_info)
    return D_star, b_star, info


def compute_ce_sensitivity_weights(
    model,
    tokens_list: List[torch.Tensor],
    layer: int,
    hook_site: str,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate per-dimension CE sensitivity of hidden activations at (layer, hook_site).

    Returns w_j = (1/N) sum_i g_{ij}^2, mean-normalized with epsilon floor.
    Gradients are used purely as a sensitivity diagnostic (no parameter updates).
    """
    from utils import _resolve_hook_name

    if device is None:
        device = Config.device

    hook_name = _resolve_hook_name(layer, hook_site)
    d_model = model.cfg.d_model

    grad_sq_sum = torch.zeros(d_model, device="cpu", dtype=torch.float64)
    n_total = 0

    for batch_start in range(0, len(tokens_list), batch_size):
        batch_end = min(batch_start + batch_size, len(tokens_list))
        for t in tokens_list[batch_start:batch_end]:
            tok = t.unsqueeze(0).to(device) if t.dim() == 1 else t.to(device)
            if tok.shape[1] < 2:
                continue

            captured_h = {}

            def _capture_hook(act, hook):
                act_out = act.clone()
                h_leaf = act[:, -2, :].clone().detach().requires_grad_(True)
                captured_h["v"] = h_leaf
                act_out[:, -2, :] = h_leaf
                return act_out

            with torch.enable_grad():
                logits = model.run_with_hooks(
                    tok, fwd_hooks=[(hook_name, _capture_hook)]
                )
                target_id = tok[0, -1]
                logit_at_pos = logits[0, -2, :]
                ce = F.cross_entropy(logit_at_pos.unsqueeze(0), target_id.unsqueeze(0))

                h = captured_h["v"]
                g = torch.autograd.grad(ce, h)[0]
                grad_sq_sum += g[0].detach().cpu().to(torch.float64).pow(2)
                n_total += 1

    if n_total == 0:
        return torch.ones(d_model, dtype=torch.float32)

    w = (grad_sq_sum / n_total).to(torch.float32)
    w = w / w.mean().clamp_min(eps)
    w = w.clamp_min(eps)

    # Power tempering: α=1.0 → full, α=0.5 → sqrt, α=0.0 → uniform
    alpha = float(getattr(Config, "GAE_CE_WEIGHTS_ALPHA", 0.5))
    if alpha != 1.0:
        w = w.pow(alpha)
        w = w / w.mean().clamp_min(eps)
        w = w.clamp_min(eps)

    return w


def extract_dictionary_from_explainer(explainer, activations, model_type='transcoder', threshold=0.01,
                                      dict_space="auto"):
    """
    Extract dictionary B from an explainer model.

    For Transcoder: B is the decoder weight matrix W_dec.T (mlp_out space).
    For SAE: B is the decoder weight matrix W_dec.T (h space).

    Args:
        explainer: Trained Transcoder or SAE model
        activations: Hidden activations, shape [N, d]
        model_type: 'transcoder' or 'sae'
        threshold: Threshold for feature activation (for empirical extraction)
        dict_space: 'auto' (default: use W_dec for both), 'decoder' (use W_dec),
                    'empirical' (legacy: least-squares h-space dictionary for transcoder)

    Returns:
        B: Dictionary matrix, shape [d, k]
    """
    explainer.eval()

    if dict_space == "auto":
        dict_space = "decoder"

    if model_type == 'transcoder':
        if dict_space == "decoder":
            # Use Transcoder's learned decoder W_dec directly (mlp_out space).
            # This respects the Transcoder's actual computation path: z → W_dec → mlp_out
            if hasattr(explainer, 'saelens') and hasattr(explainer.saelens, 'W_dec'):
                B = explainer.saelens.W_dec.T.detach().clone()  # [d_model, k]
            elif hasattr(explainer, 'W_dec'):
                B = explainer.W_dec.T.detach().clone()  # [d_model, k]
            elif hasattr(explainer, 'W_tilde'):
                B = explainer.W_tilde.detach().clone()  # [d_model, k]
            else:
                raise ValueError("Transcoder must have W_dec or W_tilde for decoder-space dictionary extraction.")
            print(f"[GAE] Using Transcoder W_dec as dictionary (mlp_out space, shape={B.shape})")
        elif dict_space == "empirical":
            # Legacy: empirical h-space dictionary via least squares (h ≈ B @ z)
            if not hasattr(explainer, 'encoder'):
                raise ValueError("Transcoder must have encoder attribute for empirical dictionary extraction.")
            device = next(explainer.parameters()).device
            dict_device = torch.device(getattr(Config, "GAE_DICT_DEVICE", str(device)))
            moved = False
            if dict_device.type == "cpu" and device.type == "cuda":
                explainer = explainer.to(dict_device)
                moved = True
            activations = activations.to(dict_device)
            with torch.no_grad():
                z_all = explainer.encoder(activations)
                z_all = F.relu(z_all)
                H = activations
                Z = z_all
                n, k = Z.shape
                ridge = float(getattr(Config, "GAE_DICT_RIDGE", 1e-8))
                method = getattr(Config, "GAE_DICT_METHOD", "auto")
                if method not in ("normal", "row", "auto"):
                    raise ValueError(f"Unknown GAE_DICT_METHOD='{method}'. Use 'normal', 'row', or 'auto'.")
                if method == "row":
                    use_row = True
                elif method == "normal":
                    use_row = False
                else:
                    use_row = n < k
                if use_row:
                    ZZt = Z @ Z.T
                    if ridge > 0:
                        ZZt = ZZt + ridge * torch.eye(n, device=ZZt.device, dtype=ZZt.dtype)
                    Y = torch.linalg.solve(ZZt, H)
                    B = Y.T @ Z
                else:
                    ZTZ = Z.T @ Z + ridge * torch.eye(k, device=Z.device, dtype=Z.dtype)
                    B = (H.T @ Z) @ torch.linalg.inv(ZTZ)
            if moved:
                explainer = explainer.to(device)
                B = B.to(device)
            print(f"[GAE] Using empirical h-space dictionary (legacy, shape={B.shape})")
        else:
            raise ValueError(f"Unknown dict_space='{dict_space}' for transcoder.")
    
    elif model_type == 'sae':
        # For SAE, prefer a direct linear decoder weight if available.
        # Check explainer.saelens.W_dec first (SaeLensSparseAutoencoderAdapter).
        if hasattr(explainer, 'W_dec'):
            B = explainer.W_dec.T  # [d, k]
        elif hasattr(explainer, 'saelens') and hasattr(explainer.saelens, 'W_dec'):
            B = explainer.saelens.W_dec.T.detach().clone()  # [d, k]
            print(f"[GAE] Using SAE W_dec directly from saelens (shape={B.shape})")
        elif hasattr(explainer, 'decoder'):
            dec = explainer.decoder
            if hasattr(dec, "weight"):
                # e.g., nn.Linear
                B = dec.weight.T  # [d, k]
            else:
                # e.g., nn.Sequential. Build an effective dictionary by decoding one-hot feature vectors:
                # For each feature j, set z = e_j and take decoded output as column b_j.
                k = getattr(explainer, "k", None)
                if k is None:
                    raise ValueError("SAE explainer must have attribute 'k' to extract dictionary from sequential decoder")
                device = next(explainer.parameters()).device
                # Use reasonably-sized chunks to avoid OOM for large k
                chunk = 128
                cols = []
                with torch.no_grad():
                    for start in range(0, k, chunk):
                        end = min(k, start + chunk)
                        z = torch.zeros(end - start, k, device=device)
                        z[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = 1.0
                        h = dec(z)  # [chunk, d]
                        cols.append(h)
                H = torch.cat(cols, dim=0)  # [k, d]
                B = H.T  # [d, k]
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return B


def apply_gae_to_explainer(
    explainer,
    id_activations,
    ood_activations,
    r=None,
    model_type='transcoder',
    rank_mode='fixed',
    rank_energy=0.99,
    rank_delta_mode='ood',
    rank_min=1,
    rank_max=None,
):
    """
    Apply GAE to an existing explainer model.
    
    Args:
        explainer: Trained Transcoder or SAE instance
        id_activations: ID activations for extracting original B, shape [N_id, d]
        ood_activations: OOD activations for computing OOD-active subspace, shape [N_ood, d]
        r: Target rank for fixed mode (default: Config.OOD_SUBSPACE_RANK_R)
        model_type: 'transcoder' or 'sae'
        rank_mode: 'fixed' (existing behavior) or 'energy' (auto rank from ΔC)
        rank_energy: cumulative squared singular-value energy target (default: 0.99)
        rank_delta_mode: 'ood' -> ΔC=Cov_ood, 'contrastive' -> ΔC=Cov_ood-Cov_id
        rank_min: lower bound for selected rank
        rank_max: optional upper bound for selected rank
    
    Returns:
        B_gae: Geometry-adapted dictionary, shape [d, k]
        B_original: Original dictionary, shape [d, k]
        D_star: Decoder-only refit on OOD, shape [d, k]
    """
    explainer.eval()

    if rank_mode not in ("fixed", "energy"):
        raise ValueError(f"Unknown rank_mode='{rank_mode}'. Use 'fixed' or 'energy'.")

    # Extract original dictionary B
    B_original = extract_dictionary_from_explainer(explainer, id_activations, model_type)
    d, k = B_original.shape

    # Ensure activations are on the same device as B
    device = B_original.device
    ood_activations = ood_activations.to(device)
    id_activations = id_activations.to(device)

    # Rank selection:
    #  - fixed: existing behavior (kept for backward compatibility)
    #  - energy: SVD/eigen energy criterion on ΔC
    if rank_mode == "fixed":
        if r is None:
            r = Config.OOD_SUBSPACE_RANK_R
        r = int(min(r, min(d, k)))
    else:
        auto_r, rank_info = select_rank_for_gae(
            B=B_original,
            H_ood=ood_activations,
            H_id=id_activations,
            energy_ratio=rank_energy,
            delta_mode=rank_delta_mode,
            min_rank=rank_min,
            max_rank=rank_max,
        )
        r = int(auto_r)
        print(
            f"[GAE] Auto rank selected from ΔC ({rank_info['delta_mode']}): "
            f"r={r}, captured={rank_info['energy_ratio_captured']:.6f}, "
            f"target={rank_info['energy_ratio_target']:.6f}, cap={rank_info['rank_cap']}"
        )

    print(f"Applying GAE with rank r={r} (d={d}, k={k}, rank_mode={rank_mode})")
    print(f"  ID activations: {id_activations.shape}")
    print(f"  OOD activations: {ood_activations.shape}")
    
    # Apply GAE
    B_gae, gae_info = gae_torch(B_original, ood_activations, r)

    # Match dictionary scale to avoid exploding reconstructions / logits
    with torch.no_grad():
        # Column-wise norm matching (feature-wise scale preservation)
        orig_norms = B_original.norm(dim=0)  # [k]
        gae_norms = B_gae.norm(dim=0)        # [k]
        scale = orig_norms / (gae_norms + 1e-8)
        B_gae = B_gae * scale.unsqueeze(0)
    
    # Decoder-only refit on OOD:
    #   - use B_gae for feature extraction
    #   - fit D_star so that h ≈ D_star @ a  on OOD activations
    with torch.no_grad():
        # Features A from B_gae (linear + ReLU, consistent with evaluation/diagnostics)
        A_ood = F.relu(ood_activations @ B_gae)  # [N_ood, k]
        D_star = decoder_only_refit(ood_activations, A_ood, lam=1e-4)  # [d, k]

    # Consistent rotation decoder (same transform as encoder, no OOD fitting)
    D_consistent = consistent_rotation_decoder(B_original, gae_info)
    with torch.no_grad():
        D_consistent = D_consistent * scale.unsqueeze(0)

    # Sanity checks
    print(f"\nGAE Sanity Checks:")
    print(f"  B_original shape: {B_original.shape}")
    print(f"  B_gae shape: {B_gae.shape}")
    assert B_gae.shape == B_original.shape, "Shape mismatch after GAE"

    return B_gae, B_original, D_star, D_consistent


@torch.no_grad()
def fit_residual_decoder(
    W_base: torch.Tensor,
    b_base: torch.Tensor,
    H: torch.Tensor,
    Z: torch.Tensor,
    lam_res: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Closed-form residual decoder fit: W_new = W_base + ΔW, b_new = b_base + Δb.

    Solves: (ΔW, Δb) = argmin (1/N)||R - ΔW·Z^T - Δb||_F^2 + lam_res||ΔW||_F^2
    where R_i = h_i - W_base @ z_i - b_base  (base decoder's residual error).

    Args:
        W_base: [d, k] base decoder weight (W_orig or W̃_rotated)
        b_base: [d] base decoder bias
        H: [N, d] OOD target activations
        Z: [N, k] encoder features (frozen)
        lam_res: ridge penalty on ΔW (controls nComp/dCE balance)

    Returns:
        W_new: [d, k] = W_base + ΔW
        b_new: [d] = b_base + Δb
        info: dict with fit statistics
    """
    N, d = H.shape
    k = Z.shape[1]
    device = H.device
    dtype = H.dtype

    # Residuals: r_i = h_i - W_base @ z_i - b_base
    R = H - Z @ W_base.T - b_base.unsqueeze(0)  # [N, d]

    # Center
    z_mean = Z.mean(dim=0)        # [k]
    r_mean = R.mean(dim=0)        # [d]
    Zc = Z - z_mean.unsqueeze(0)  # [N, k]
    Rc = R - r_mean.unsqueeze(0)  # [N, d]

    # Σ_zz = (1/N) Zc^T Zc + λ_res I   — [k, k]
    Sigma_zz = (Zc.T @ Zc) / N + lam_res * torch.eye(k, device=device, dtype=dtype)

    # Σ_rz = (1/N) Rc^T Zc             — [d, k]
    Sigma_rz = (Rc.T @ Zc) / N

    # ΔW = Σ_rz @ Σ_zz⁻¹  — solve via Cholesky for numerical stability
    # Σ_zz @ ΔW^T = Σ_rz^T  →  ΔW^T = Σ_zz⁻¹ @ Σ_rz^T
    L = torch.linalg.cholesky(Sigma_zz)
    delta_W_T = torch.cholesky_solve(Sigma_rz.T, L)  # [k, d]
    delta_W = delta_W_T.T  # [d, k]

    # Δb = r_mean - ΔW @ z_mean
    delta_b = r_mean - delta_W @ z_mean

    W_new = W_base + delta_W
    b_new = b_base + delta_b

    # Fit statistics
    H_hat = Z @ W_new.T + b_new.unsqueeze(0)
    mse = float(torch.mean((H - H_hat).pow(2)).item())
    base_mse = float(torch.mean(R.pow(2)).item())
    delta_W_norm = float(delta_W.norm().item())
    W_base_norm = float(W_base.norm().item())

    info = {
        "lam_res": lam_res,
        "fit_mse": mse,
        "base_mse": base_mse,
        "delta_W_norm": delta_W_norm,
        "delta_W_ratio": delta_W_norm / max(W_base_norm, 1e-8),
        "fit_samples": N,
    }
    return W_new, b_new, info


@torch.no_grad()
def fit_selective_decoder_columns(
    W_dec: torch.Tensor,   # [k, d] original decoder
    b_dec: torch.Tensor,   # [d]
    H: torch.Tensor,       # [N, d] OOD target activations
    Z: torch.Tensor,       # [N, k] encoder features (frozen, sparse)
    column_indices: torch.Tensor,  # [n_sel] indices of columns to update
    lam_res: float = 0.1,
):
    """
    SAE-only Option B: update only the specified decoder columns via residual fit.
    Other columns remain at W_dec.

    Math: for j in column_indices:
      ΔW[:, j] = closed-form ridge regression on (Z[:, column_indices], R)
      where R = H - Z @ W_dec.T - b_dec is the original residual.
      W_new[:, j] = W_dec[:, j] + ΔW[:, j]   for j in column_indices
      W_new[:, j] = W_dec[:, j]              otherwise
    """
    N, d = H.shape
    k_total = Z.shape[1]
    n_sel = int(column_indices.numel())
    device = H.device
    dtype = H.dtype

    # Original decoder residual
    R = H - Z @ W_dec - b_dec.unsqueeze(0)  # [N, d]   (W_dec shape [k,d] used directly: Z @ W_dec)

    # Subset Z to selected columns
    Z_sel = Z[:, column_indices]  # [N, n_sel]

    # Center
    z_mean = Z_sel.mean(dim=0)
    r_mean = R.mean(dim=0)
    Zc = Z_sel - z_mean.unsqueeze(0)
    Rc = R - r_mean.unsqueeze(0)

    # Closed-form: ΔW_sel = Σ_rz @ Σ_zz⁻¹  (n_sel × d via solving Σ_zz @ ΔW_sel.T = Σ_rz.T)
    Sigma_zz = (Zc.T @ Zc) / N + lam_res * torch.eye(n_sel, device=device, dtype=dtype)
    Sigma_rz = (Rc.T @ Zc) / N  # [d, n_sel]
    L = torch.linalg.cholesky(Sigma_zz)
    delta_W_sel_T = torch.cholesky_solve(Sigma_rz.T, L)  # [n_sel, d]
    delta_W_sel = delta_W_sel_T  # [n_sel, d] (W_dec is [k, d] convention)

    # Bias correction (for the selected columns only)
    delta_b = r_mean - delta_W_sel.T @ z_mean  # [d]

    # Build W_new: copy W_dec, update only selected columns
    W_new = W_dec.clone()
    W_new[column_indices] = W_dec[column_indices] + delta_W_sel
    b_new = b_dec + delta_b

    H_hat = Z @ W_new + b_new.unsqueeze(0)
    mse = float(torch.mean((H - H_hat).pow(2)).item())
    base_mse = float(torch.mean(R.pow(2)).item())
    delta_W_norm = float(delta_W_sel.norm().item())

    info = {
        "lam_res": lam_res,
        "fit_mse": mse,
        "base_mse": base_mse,
        "n_columns_updated": n_sel,
        "n_columns_total": k_total,
        "frac_columns_updated": n_sel / max(k_total, 1),
        "delta_W_sel_norm": delta_W_norm,
        "fit_samples": N,
    }
    return W_new, b_new, info


@torch.no_grad()
def apply_gae_encoder_rotation(
    saelens_model,
    U_ood: torch.Tensor,
    U_ref: torch.Tensor,
    R_align: torch.Tensor,
    preserve_orthogonal: bool = True,
    mix: float = 1.0,
):
    """
    SAE-only: apply Procrustes rotation to encoder weights, keeping decoder fixed.

    Rationale:
      Top-K SAE decoder columns are highly specialized. Modifying them (Step 2 of
      Algorithm 1) destroys downstream CE. Apply the same Procrustes rotation R*
      computed from the OOD subspace to the *encoder* instead, so the encoder
      activates features whose decoder direction better aligns with OOD energy.

    Math:
      R_input = U_ood @ R_align @ U_ref^T          # rank-r operator
      M = R_input + (I - U_ref U_ref^T)            # rotate inside U_ref, identity outside
      W_enc_new = M @ W_enc                         # apply to encoder columns

    Args:
      saelens_model: SAELens SparseAutoencoder (encoder weights modified in place).
      U_ood:    [d, r] OOD top-r eigenvectors.
      U_ref:    [d, r] decoder column subspace (top-r left SVD of B).
      R_align:  [r, r] Procrustes alignment matrix.
      preserve_orthogonal: if True, keep encoder structure outside U_ref (recommended).
      mix:      0.0=no rotation (Fixed), 1.0=full rotation.

    Returns:
      info dict with stats. (encoder weights modified in place; caller should
      reload SAE if multiple configs need to share a process.)
    """
    device = saelens_model.W_enc.device
    dtype = saelens_model.W_enc.dtype
    U_ood_d = U_ood.to(device=device, dtype=dtype)
    U_ref_d = U_ref.to(device=device, dtype=dtype)
    R_d = R_align.to(device=device, dtype=dtype)

    R_input = U_ood_d @ R_d @ U_ref_d.T  # [d, d], rank-r

    if preserve_orthogonal:
        d = U_ref_d.shape[0]
        I_d = torch.eye(d, device=device, dtype=dtype)
        proj_ref = U_ref_d @ U_ref_d.T  # [d, d]
        M = R_input + (I_d - proj_ref)
    else:
        M = R_input

    mix_eff = float(max(0.0, min(1.0, mix)))
    W_enc_orig = saelens_model.W_enc.data.clone()
    W_enc_rotated = M @ W_enc_orig  # [d, k]
    W_enc_new = (1.0 - mix_eff) * W_enc_orig + mix_eff * W_enc_rotated

    saelens_model.W_enc.data = W_enc_new

    info = {
        "preserve_orthogonal": bool(preserve_orthogonal),
        "mix": mix_eff,
        "rank_r": int(U_ref_d.shape[1]),
        "rotation_matrix_norm": float(R_input.norm().item()),
        "delta_W_enc_norm": float((W_enc_new - W_enc_orig).norm().item()),
        "W_enc_orig_norm": float(W_enc_orig.norm().item()),
        "delta_ratio": float((W_enc_new - W_enc_orig).norm().item() / max(W_enc_orig.norm().item(), 1e-8)),
    }
    return info


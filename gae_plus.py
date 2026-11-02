"""GAE+: choose how much of GAE's Step 1 and Step 2 changes to apply.

A = [W, b] stacks decoder and bias (shape [d, k+1]); z_bar = [z; 1].
A_plus = A0 + eta1 (A1 - A0) + eta2 (A2 - A1),  0 <= eta2 <= eta1 <= 1.
"""
import math

import torch


def eta_objective_terms(Z, Y, A0, A1, A2):
    """Z [N, k], Y [N, d], A_j [d, k+1] -> H [2, 2], g [2], K [2, 2], R0 (float)."""
    Zb = torch.cat([Z, Z.new_ones(Z.shape[0], 1)], dim=1)  # z_bar_i = [z_i; 1], [N, k+1]
    D = torch.stack([A1 - A0, A2 - A1])                    # D_1, D_2, [2, d, k+1]
    E0 = Y - Zb @ A0.T                                     # e0_i = y_i - A0 z_bar_i, [N, d]
    P = torch.einsum('nk,jdk->jnd', Zb, D)                 # P[j, i] = D_j z_bar_i, [2, N, d]
    N = Z.shape[0]
    H = torch.einsum('jnd,lnd->jl', P, P) / N              # H_jl = mean_i <D_j z_bar_i, D_l z_bar_i>
    g = torch.einsum('nd,jnd->j', E0, P) / N               # g_j = mean_i <e0_i, D_j z_bar_i>
    K = torch.einsum('jdk,ldk->jl', D, D)                  # K_jl = <D_j, D_l>_F
    R0 = (E0 ** 2).sum(dim=1).mean().item()                # R0 = mean_i ||e0_i||^2 (= d * fit_mse)
    return H, g, K, R0


def bernstein_radius(L, v, dim, delta):
    """Tropp (2015) Thm 1.6.2 solved for t: P(||Z|| >= t) <= delta."""
    ell = math.log(dim / delta)
    return L * ell / 3 + math.sqrt((L * ell / 3) ** 2 + 2 * v * ell)


def eb_radius(V_norm, dim, n, alpha):
    """Wang & Ramdas (arXiv 2411.09516) Thm 3.1, Eq. (17): D_n for [0,1]-scaled matrices."""
    ell = math.log(n * dim / ((n - 1) * alpha))
    return (ell / (3 * n) + math.sqrt(2 * V_norm * ell / n)
            + (math.sqrt(5 / 3) + 1) * math.sqrt(ell * math.log(2 * n * dim / alpha)) / n)


def paired_variance_norms(Z, E0, B, R):
    """||V*_n|| for X_i = z_bar z_bar^T / B and for rescaled dilations of vec(e0_i z_bar_i^T)."""
    n = Z.shape[0] // 2 * 2                                # even n (drop the last sample if odd)
    Zb = torch.cat([Z[:n], Z.new_ones(n, 1)], dim=1)       # z_bar_i, [n, k+1]
    a, b = Zb[0::2], Zb[1::2]                              # pairs (z_bar_1, z_bar_2), (z_bar_3, z_bar_4), ...
    ea, eb = E0[:n:2], E0[1:n:2]                           # matching e0 pairs
    aa, bb, ab = (a * a).sum(1), (b * b).sum(1), (a * b).sum(1)
    C = torch.stack([torch.stack([aa, -ab], -1), torch.stack([-ab, bb], -1)], -2)   # C_j, [n/2, 2, 2]
    w, Q = torch.linalg.eigh(C)
    S = torch.block_diag(*(Q * w.clamp_min(0).sqrt()[:, None, :]) @ Q.transpose(1, 2))  # C^{1/2}, [n, n]
    vM = torch.linalg.eigvalsh(S @ (Zb @ Zb.T) @ S)[-1].item() / (n * B * B)
    vT = ((ea * ea).sum(1) * aa + (eb * eb).sum(1) * bb
          - 2 * (ea * eb).sum(1) * ab).sum().item() / (4 * R * R * n)
    return vM, vT


def compute_bounds(Z, E0, delta=0.05, method="A"):
    """a_N, b_N with observed ranges (no guarantee for the ranges). method: "A" | "B"."""
    N, k = Z.shape
    d = E0.shape[1]
    B = ((Z ** 2).sum(dim=1) + 1).max().item()             # max_i ||z_bar_i||^2 (observed)
    E = E0.norm(dim=1).max().item()                        # max_i ||e0_i|| (observed)
    assert method in ("A", "B"), method
    if method == "A":                                      # Tropp (2015) Thm 1.6.2
        a_N = bernstein_radius(2 * B / N, B * B / N, 2 * (k + 1), delta / 2)
        b_N = bernstein_radius(2 * E * math.sqrt(B) / N, E * E * B / N, d * (k + 1) + 1, delta / 2)
    else:                                                  # Wang & Ramdas Thm 3.1
        R = E * math.sqrt(B)                               # range of the dilation eigenvalues
        vM, vT = paired_variance_norms(Z, E0, B, R)
        n = N // 2 * 2
        a_N = B * eb_radius(vM, k + 1, n, delta / 4)       # X_i and I - X_i -> two-sided
        b_N = 2 * R * eb_radius(vT, d * (k + 1) + 1, n, delta / 2)
    return a_N, b_N


def select_eta(H, g, K, a_N=0.0, b_N=0.0, step=0.01):
    """Grid search of J over 0 <= eta2 <= eta1 <= 1 -> (eta1, eta2, J_min)."""
    t = torch.arange(0, round(1 / step) + 1, dtype=H.dtype, device=H.device) * step  # 0, 0.01, ..., 1
    X = torch.cartesian_prod(t, t)                                       # all (eta1, eta2) pairs
    X = X[X[:, 1] <= X[:, 0] + 1e-12]                                    # keep 0 <= eta2 <= eta1
    xKx = torch.einsum('pj,jl,pl->p', X, K, X)                           # ||D(x)||_F^2 per point
    J = torch.einsum('pj,jl,pl->p', X, H, X) - 2 * X @ g \
        + a_N * xKx + 2 * b_N * xKx.clamp_min(0).sqrt()                  # J(x) per point
    i = torch.argmin(J)
    return X[i, 0].item(), X[i, 1].item(), J[i].item()


def apply_eta(A0, A1, A2, eta1, eta2):
    """A_plus = A0 + eta1 (A1 - A0) + eta2 (A2 - A1)."""
    return A0 + eta1 * (A1 - A0) + eta2 * (A2 - A1)

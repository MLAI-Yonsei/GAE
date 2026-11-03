"""GAE+: choose how much of GAE's Step 1 and Step 2 changes to apply.

A = [W, b] stacks decoder and bias (shape [d, k+1]); z_bar = [z; 1].
A_plus = A0 + eta1 (A1 - A0) + eta2 (A2 - A1),  0 <= eta2 <= eta1 <= 1.

Algorithm (entry point: gae_plus_select):
  1. Split the OOD samples into halves A and B.
  2. On A, run GAE: A1 = Step 1 (Eqs. 5-6), A2 = Step 2 (Eqs. 8-11, lambda_id = 0); A0 = original.
  3. On B (held out), estimate the change in reconstruction error and its uncertainty:
     mode "elementwise" (default): per-entry bounds on H, g in the (A1 - A0, A2 - A0) basis;
     mode "tie": eta1 = eta2, bounds on the single A2 - A0 direction;
     mode "matrix": free eta, operator-norm bound on the 2x2 H (step basis).
     bound "best" (default): min of Chebyshev and Bernstein-type radii; "bernstein": Tropp / Pinelis only.
  4. eta = argmin over the grid of (sample error change + uncertainty penalty); return A_plus.
Steps 1-2 (data, GAE) live in gae_plus_probe.build_candidates / gae_plus_bounds_grid.py.

Original plan (methodology draft Sec. 3-4; no sample split, uniform bounds, K = ||D||_F^2):
  gae_plus_bounds_grid.py --split off, i.e. eta_objective_terms -> compute_bounds(method=...) ->
  select_eta(H, g, K_D, a_N, b_N); method "A" = Sec. 4.2 with c from Tropp's matrix Bernstein,
  "chebyshev" = c = sqrt(2 / (N delta)), "B" = Sec. 4.2 variance-based follow-up.
  gae_plus_select (split on) is not part of the original plan.
"""
import math

import torch


def _per_sample(Z, Y, A0, A1, A2, basis="step"):
    """D [2, d, k+1], E0 [N, d] (e0_i = y_i - A0 z_bar_i), P [2, N, d] (P[j, i] = D_j z_bar_i).
    basis "step": D = (A1 - A0, A2 - A1); "gae": D = (A1 - A0, A2 - A0)."""
    Zb = torch.cat([Z, Z.new_ones(Z.shape[0], 1)], dim=1)  # z_bar_i = [z_i; 1], [N, k+1]
    D = torch.stack([A1 - A0, (A2 - A1) if basis == "step" else (A2 - A0)])  # [2, d, k+1]
    E0 = Y - Zb @ A0.T                                     # e0_i = y_i - A0 z_bar_i, [N, d]
    P = torch.einsum('nk,jdk->jnd', Zb, D)                 # P[j, i] = D_j z_bar_i, [2, N, d]
    return D, E0, P


def eta_objective_terms(Z, Y, A0, A1, A2):
    """Z [N, k], Y [N, d], A_j [d, k+1] -> H [2, 2], g [2], K [2, 2], R0 (float)."""
    D, E0, P = _per_sample(Z, Y, A0, A1, A2)
    N = Z.shape[0]
    H = torch.einsum('jnd,lnd->jl', P, P) / N              # H_jl = mean_i <D_j z_bar_i, D_l z_bar_i>
    g = torch.einsum('nd,jnd->j', E0, P) / N               # g_j = mean_i <e0_i, D_j z_bar_i>
    K = torch.einsum('jdk,ldk->jl', D, D)                  # K_jl = <D_j, D_l>_F
    R0 = (E0 ** 2).sum(dim=1).mean().item()                # R0 = mean_i ||e0_i||^2 (= d * fit_mse)
    return H, g, K, R0


def chebyshev_radius(s, n, delta):
    """Chebyshev: P(||mean - mu|| >= t) <= s^2 / (n t^2), s^2 >= E||X||^2 -> t = s / sqrt(n delta)."""
    return s / math.sqrt(n * delta)


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


def compute_bounds(Z, E0, delta=0.05, method="best"):
    """Split off: a_N >= ||M_hat - M||_op, b_N >= ||T_hat - T||_F with observed ranges
    (no guarantee for the ranges). method: "A" | "B" | "chebyshev" | "bernstein" | "best".
    "bernstein": a_N Tropp ambient, b_N Pinelis; "best": min(chebyshev, bernstein) per bound."""
    N, k = Z.shape
    d = E0.shape[1]
    B = ((Z ** 2).sum(dim=1) + 1).max().item()             # max_i ||z_bar_i||^2 (observed)
    E = E0.norm(dim=1).max().item()                        # max_i ||e0_i|| (observed)
    assert method in ("A", "B", "chebyshev", "bernstein", "best"), method
    a_cheb = chebyshev_radius(B, N, delta / 2)             # E||z z^T - M||_F^2 <= B^2
    b_cheb = chebyshev_radius(E * math.sqrt(B), N, delta / 2)  # E||e z^T - T||_F^2 <= E^2 B
    if method == "chebyshev":
        return a_cheb, b_cheb
    if method in ("A", "bernstein", "best"):               # Tropp (2015) Thm 1.6.2
        a_N = bernstein_radius(2 * B / N, B * B / N, 2 * (k + 1), delta / 2)
        b_N = bernstein_radius(2 * E * math.sqrt(B) / N, E * E * B / N, d * (k + 1) + 1, delta / 2)
    if method in ("bernstein", "best"):                    # b_N: Pinelis, dimension-free (dim = 2)
        b_pin = bernstein_radius(2 * E * math.sqrt(B) / N, E * E * B / N, 2, delta / 2)
        if method == "bernstein":
            return a_N, b_pin
        return min(a_cheb, a_N), min(b_cheb, b_pin)
    if method == "B":                                      # Wang & Ramdas Thm 3.1
        R = E * math.sqrt(B)                               # range of the dilation eigenvalues
        vM, vT = paired_variance_norms(Z, E0, B, R)
        n = N // 2 * 2
        a_N = B * eb_radius(vM, k + 1, n, delta / 4)       # X_i and I - X_i -> two-sided
        b_N = 2 * R * eb_radius(vT, d * (k + 1) + 1, n, delta / 2)
    return a_N, b_N


def split_bounds(Z, Y, A0, A1, A2, delta=0.05, method="best"):
    """Split on (Z, Y = held-out samples, candidates built elsewhere; use K = I):
    a_N >= ||H_hat - H||_op (2x2), b_N >= ||g_hat - g||_2.
    method: "chebyshev" | "bernstein" (Tropp 2x2 / Pinelis) | "best" (min of the two)."""
    assert method in ("chebyshev", "bernstein", "best"), method
    _, E0, P = _per_sample(Z, Y, A0, A1, A2)
    n = Z.shape[0]
    L_H = (P ** 2).sum(dim=(0, 2)).max().item()            # max_i tr Q_i >= ||Q_i||_op, ||Q_i||_F
    L_g = torch.einsum('nd,jnd->nj', E0, P).norm(dim=1).max().item()  # max_i ||q_i||
    a_cheb = chebyshev_radius(L_H, n, delta / 2)
    b_cheb = chebyshev_radius(L_g, n, delta / 2)
    if method == "chebyshev":
        return a_cheb, b_cheb
    a_bern = bernstein_radius(2 * L_H / n, L_H * L_H / n, 4, delta / 2)  # Tropp Thm 1.6.2, 2x2
    b_pin = bernstein_radius(2 * L_g / n, L_g * L_g / n, 2, delta / 2)   # Pinelis, dimension-free
    if method == "bernstein":
        return a_bern, b_pin
    return min(a_cheb, a_bern), min(b_cheb, b_pin)


def elementwise_bounds(Z, Y, A0, A1, A2, delta=0.05, method="best", basis="step"):
    """Split on, Bonferroni over the 5 numbers (H11, H12, H22, g1, g2), delta/5 each, in `basis`.
    method: "chebyshev" | "bernstein" (scalar Bernstein) | "best" (min of the two).
    Returns H, g (held-out means) and per-entry radii rH [2, 2], rg [2] (observed ranges)."""
    assert method in ("chebyshev", "bernstein", "best"), method
    _, E0, P = _per_sample(Z, Y, A0, A1, A2, basis)
    n, dl = Z.shape[0], delta / 5
    Q = torch.einsum('jnd,lnd->njl', P, P)                 # per-sample 2x2
    q = torch.einsum('nd,jnd->nj', E0, P)                  # per-sample 2
    s = torch.cat([Q.abs().amax(0).flatten(), q.abs().amax(0)]).tolist()  # max_i |entry| (observed)
    r_cheb = [chebyshev_radius(x, n, dl) for x in s]
    r_bern = [bernstein_radius(2 * x / n, x * x / n, 2, dl) for x in s]  # scalar Bernstein (d1 + d2 = 2)
    r = {"chebyshev": r_cheb, "bernstein": r_bern, "best": [min(c, b) for c, b in zip(r_cheb, r_bern)]}[method]
    rH, rg = torch.tensor(r[:4], dtype=Q.dtype, device=Q.device).view(2, 2), torch.tensor(r[4:], dtype=Q.dtype, device=Q.device)
    return Q.mean(0), q.mean(0), rH, rg


def _eta_grid(step, dtype, device):
    """All (eta1, eta2) on a step grid with 0 <= eta2 <= eta1 <= 1."""
    t = torch.arange(0, round(1 / step) + 1, dtype=dtype, device=device) * step
    X = torch.cartesian_prod(t, t)
    return X[X[:, 1] <= X[:, 0] + 1e-12]


def select_eta_elementwise(H, g, rH, rg, basis="step", step=0.01):
    """Grid search of J = y^T H y - 2 g^T y + sum_jl |y_j||y_l| rH_jl + 2 sum_j |y_j| rg_j,
    y = (eta1, eta2) ["step"] or (eta1 - eta2, eta2) ["gae"] -> (eta1, eta2, J_min)."""
    X = _eta_grid(step, H.dtype, H.device)
    Yc = X if basis == "step" else torch.stack([X[:, 0] - X[:, 1], X[:, 1]], dim=1)
    Ya = Yc.abs()
    J = torch.einsum('pj,jl,pl->p', Yc, H, Yc) - 2 * Yc @ g \
        + torch.einsum('pj,jl,pl->p', Ya, rH, Ya) + 2 * Ya @ rg
    i = torch.argmin(J)
    return X[i, 0].item(), X[i, 1].item(), J[i].item()


def select_eta(H, g, K, a_N=0.0, b_N=0.0, step=0.01):
    """Grid search of J over 0 <= eta2 <= eta1 <= 1 -> (eta1, eta2, J_min)."""
    X = _eta_grid(step, H.dtype, H.device)                               # 0 <= eta2 <= eta1 <= 1
    xKx = torch.einsum('pj,jl,pl->p', X, K, X)                           # ||D(x)||_F^2 per point
    J = torch.einsum('pj,jl,pl->p', X, H, X) - 2 * X @ g \
        + a_N * xKx + 2 * b_N * xKx.clamp_min(0).sqrt()                  # J(x) per point
    i = torch.argmin(J)
    return X[i, 0].item(), X[i, 1].item(), J[i].item()


def apply_eta(A0, A1, A2, eta1, eta2):
    """A_plus = A0 + eta1 (A1 - A0) + eta2 (A2 - A1)."""
    return A0 + eta1 * (A1 - A0) + eta2 * (A2 - A1)


def gae_plus_select(Z, Y, A0, A1, A2, mode="elementwise", bound="best", delta=0.05):
    """GAE+ selection on held-out samples Z, Y (A1, A2 must be built on other samples).
    mode "elementwise": Bonferroni bounds on (H11, H12, H22, g1, g2) in the (A1 - A0, A2 - A0) basis, free eta.
    mode "tie": eta1 = eta2 = t (A_plus = A0 + t (A2 - A0)), 2x2 bounds with K = diag(0, 1).
    mode "matrix": free eta, one operator-norm bound on the 2x2 H and one on g (step basis), K = I.
    bound: "best" (min of Chebyshev and Bernstein-type, default) | "bernstein" (Tropp / Pinelis) | "chebyshev".
    Returns eta1, eta2, A_plus."""
    assert mode in ("elementwise", "tie", "matrix"), mode
    assert bound in ("best", "bernstein", "chebyshev"), bound
    if mode == "elementwise":
        H, g, rH, rg = elementwise_bounds(Z, Y, A0, A1, A2, delta, bound, basis="gae")
        e1, e2, _ = select_eta_elementwise(H, g, rH, rg, basis="gae")
        return e1, e2, apply_eta(A0, A1, A2, e1, e2)
    if mode == "matrix":
        H, g, _, _ = eta_objective_terms(Z, Y, A0, A1, A2)
        a_N, b_N = split_bounds(Z, Y, A0, A1, A2, delta, bound)
        e1, e2, _ = select_eta(H, g, torch.eye(2, dtype=H.dtype, device=H.device), a_N, b_N)
        return e1, e2, apply_eta(A0, A1, A2, e1, e2)
    H, g, _, _ = eta_objective_terms(Z, Y, A0, A0, A2)     # A1 := A0 -> only the A2 - A0 direction
    a_N, b_N = split_bounds(Z, Y, A0, A0, A2, delta, bound)
    K = torch.diag(torch.tensor([0.0, 1.0], dtype=H.dtype, device=H.device))
    e1, e2, _ = select_eta(H, g, K, a_N, b_N)
    return e1, e2, apply_eta(A0, A0, A2, e1, e2)

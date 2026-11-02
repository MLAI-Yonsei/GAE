"""Checks for gae_plus.py. Run: python tests/test_gae_plus.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import math

from gae_plus import (apply_eta, bernstein_radius, compute_bounds, eb_radius,
                      eta_objective_terms, paired_variance_norms, select_eta)

torch.manual_seed(0)
DT = torch.float64


def _random_problem(N=50, k=7, d=5):
    Z = torch.relu(torch.randn(N, k, dtype=DT))
    Y = torch.randn(N, d, dtype=DT)
    A0, A1, A2 = (torch.randn(d, k + 1, dtype=DT) for _ in range(3))
    return Z, Y, A0, A1, A2


def _recon_err(Z, Y, A):
    """Independent reference: mean_i ||y_i - W z_i - b||^2 with A = [W, b]."""
    W, b = A[:, :-1], A[:, -1]
    return ((Y - Z @ W.T - b) ** 2).sum(dim=1).mean().item()


def _feasible_points():
    return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.3, 0.1), (0.7, 0.7), (0.5, 0.0)]


def test_apply_eta_endpoints():
    _, _, A0, A1, A2 = _random_problem()
    assert torch.allclose(apply_eta(A0, A1, A2, 0.0, 0.0), A0)
    assert torch.allclose(apply_eta(A0, A1, A2, 1.0, 0.0), A1)
    assert torch.allclose(apply_eta(A0, A1, A2, 1.0, 1.0), A2)


def test_quadratic_form_matches_direct_error():
    Z, Y, A0, A1, A2 = _random_problem()
    H, g, K, R0 = eta_objective_terms(Z, Y, A0, A1, A2)
    assert abs(R0 - _recon_err(Z, Y, A0)) < 1e-9
    for e1, e2 in _feasible_points():
        x = torch.tensor([e1, e2], dtype=DT)
        direct = _recon_err(Z, Y, apply_eta(A0, A1, A2, e1, e2)) - R0
        assert abs((x @ H @ x - 2 * g @ x).item() - direct) < 1e-9
        D = apply_eta(A0, A1, A2, e1, e2) - A0
        assert abs((x @ K @ x).item() - (D ** 2).sum().item()) < 1e-9


def test_terms_are_symmetric_psd():
    Z, Y, A0, A1, A2 = _random_problem()
    H, _, K, _ = eta_objective_terms(Z, Y, A0, A1, A2)
    for M in (H, K):
        assert M.shape == (2, 2)
        assert torch.allclose(M, M.T)
        assert torch.linalg.eigvalsh(M).min() > -1e-12


def test_select_eta_stays_in_triangle_and_hits_corners():
    I = torch.eye(2, dtype=DT)
    # no gain anywhere -> keep the original explainer
    e1, e2, J = select_eta(I, torch.zeros(2, dtype=DT), I)
    assert (e1, e2) == (0.0, 0.0) and J == 0.0
    # unconstrained minimum at (2, 2) -> clipped to the (1, 1) corner
    e1, e2, _ = select_eta(I, torch.tensor([2.0, 2.0], dtype=DT), I)
    assert (e1, e2) == (1.0, 1.0)
    # unconstrained minimum at (2, -1) -> (1, 0) corner (Step 1 only)
    e1, e2, _ = select_eta(I, torch.tensor([2.0, -1.0], dtype=DT), I)
    assert (e1, e2) == (1.0, 0.0)
    # random problems: always feasible
    for _ in range(20):
        Z, Y, A0, A1, A2 = _random_problem()
        H, g, K, _ = eta_objective_terms(Z, Y, A0, A1, A2)
        e1, e2, _ = select_eta(H, g, K, a_N=0.1, b_N=0.1)
        assert 0.0 <= e2 <= e1 <= 1.0


def test_select_eta_without_penalty_matches_brute_force():
    Z, Y, A0, A1, A2 = _random_problem()
    H, g, K, R0 = eta_objective_terms(Z, Y, A0, A1, A2)
    e1, e2, J = select_eta(H, g, K, a_N=0.0, b_N=0.0, step=0.05)
    best = min(
        (_recon_err(Z, Y, apply_eta(A0, A1, A2, i / 20, j / 20)) - R0, i / 20, j / 20)
        for i in range(21) for j in range(i + 1)
    )
    assert abs(J - best[0]) < 1e-9
    assert abs(e1 - best[1]) < 1e-9 and abs(e2 - best[2]) < 1e-9


def test_larger_penalty_never_increases_change():
    Z, Y, A0, A1, A2 = _random_problem()
    H, g, K, _ = eta_objective_terms(Z, Y, A0, A1, A2)
    sizes = []
    for b in (0.0, 0.01, 0.1, 1.0, 10.0, 1e6):
        e1, e2, _ = select_eta(H, g, K, a_N=0.0, b_N=b)
        x = torch.tensor([e1, e2], dtype=DT)
        sizes.append((x @ K @ x).item())
    assert all(s2 <= s1 + 1e-9 for s1, s2 in zip(sizes, sizes[1:]))
    assert sizes[-1] == 0.0  # huge penalty -> original explainer


def test_bernstein_radius_solves_tail_equation():
    for L, v, dim, delta in [(0.1, 0.5, 10, 0.05), (2.0, 0.01, 49154, 0.025), (1e-3, 1e-4, 3, 0.5)]:
        t = bernstein_radius(L, v, dim, delta)
        assert t > 0
        tail = dim * math.exp(-(t * t / 2) / (v + L * t / 3))
        assert abs(tail - delta) / delta < 1e-9


def test_bernstein_radius_limits():
    ell = math.log(10 / 0.05)
    assert abs(bernstein_radius(0.0, 0.5, 10, 0.05) - math.sqrt(2 * 0.5 * ell)) < 1e-12
    assert abs(bernstein_radius(0.3, 0.0, 10, 0.05) - 2 * 0.3 * ell / 3) < 1e-12


def test_bounds_A_match_formula():
    N, k, d, delta = 40, 6, 3, 0.05
    Z = torch.relu(torch.randn(N, k, dtype=DT))
    E0 = torch.randn(N, d, dtype=DT)
    a_N, b_N = compute_bounds(Z, E0, delta)
    B = ((Z ** 2).sum(dim=1) + 1).max().item()   # max ||z_bar||^2
    E = E0.norm(dim=1).max().item()              # max ||e0||

    def t_star(L, v, dim, dl):                   # independent re-derivation
        ell = math.log(dim / dl)
        return L * ell / 3 + math.sqrt((L * ell / 3) ** 2 + 2 * v * ell)

    assert abs(a_N - t_star(2 * B / N, B * B / N, 2 * (k + 1), delta / 2)) < 1e-12
    assert abs(b_N - t_star(2 * E * math.sqrt(B) / N, E * E * B / N, d * (k + 1) + 1, delta / 2)) < 1e-12


def test_bounds_A_scaling_and_sample_size():
    Z = torch.relu(torch.randn(30, 5, dtype=DT))
    E0 = torch.randn(30, 4, dtype=DT)
    a1, b1 = compute_bounds(Z, E0)
    a2, b2 = compute_bounds(Z, 3 * E0)
    assert abs(a2 - a1) < 1e-12 and abs(b2 - 3 * b1) < 1e-9   # a_N ignores E0, b_N linear in it
    a4, b4 = compute_bounds(Z.repeat(4, 1), E0.repeat(4, 1))  # same ranges, 4x samples
    assert a4 < a1 and b4 < b1


def test_bounds_A_cover_true_error():
    """Finite population (exact M, T): the bound must fail in at most ~delta of draws."""
    gen = torch.Generator().manual_seed(1)
    m, k, d, n, delta, trials = 6, 3, 2, 200, 0.05, 300
    pop_z = torch.relu(torch.randn(m, k, dtype=DT, generator=gen))
    pop_e = torch.randn(m, d, dtype=DT, generator=gen)
    pop_zb = torch.cat([pop_z, torch.ones(m, 1, dtype=DT)], dim=1)
    M = pop_zb.T @ pop_zb / m
    T = pop_e.T @ pop_zb / m
    fails = 0
    for _ in range(trials):
        idx = torch.randint(0, m, (n,), generator=gen)
        Z, E0, Zb = pop_z[idx], pop_e[idx], pop_zb[idx]
        a_N, b_N = compute_bounds(Z, E0, delta)
        err_M = torch.linalg.matrix_norm(Zb.T @ Zb / n - M, ord=2).item()
        err_T = (E0.T @ Zb / n - T).norm().item()
        fails += (err_M > a_N) or (err_T > b_N)
    assert fails / trials <= delta


def test_eb_radius_matches_wang_ramdas_eq17():
    for V, dim, n, alpha in [(0.01, 8, 100, 0.025), (0.2, 49154, 2048, 0.0125), (0.0, 3, 10, 0.5)]:
        ell = math.log(n * dim / ((n - 1) * alpha))
        ref = (ell / (3 * n) + math.sqrt(2 * V * ell / n)
               + (math.sqrt(5 / 3) + 1) * math.sqrt(ell * math.log(2 * n * dim / alpha)) / n)
        assert abs(eb_radius(V, dim, n, alpha) - ref) < 1e-12


def test_paired_variance_norms_match_explicit_matrices():
    N, k, d = 10, 4, 3
    Z = torch.relu(torch.randn(N, k, dtype=DT))
    E0 = torch.randn(N, d, dtype=DT)
    Zb = torch.cat([Z, torch.ones(N, 1, dtype=DT)], dim=1)
    B = (Zb ** 2).sum(dim=1).max().item()
    R = E0.norm(dim=1).max().item() * math.sqrt(B)
    vM, vT = paired_variance_norms(Z, E0, B, R)
    # explicit V* for X_i = z_bar z_bar^T / B
    X = torch.einsum('ni,nj->nij', Zb, Zb) / B
    V = sum((X[a] - X[a + 1]) @ (X[a] - X[a + 1]) for a in range(0, N, 2)) / N
    assert abs(vM - torch.linalg.eigvalsh(V).abs().max().item()) < 1e-10
    # explicit V* for rescaled dilations of x_i = vec(e0_i z_bar_i^T)
    x = torch.einsum('nd,nk->ndk', E0, Zb).reshape(N, -1)
    p = x.shape[1]

    def dil(u):
        top = torch.cat([torch.zeros(p, p, dtype=DT), u[:, None]], dim=1)
        bot = torch.cat([u[None, :], torch.zeros(1, 1, dtype=DT)], dim=1)
        return torch.cat([top, bot], dim=0)

    Vd = sum((dil(x[a] - x[a + 1]) / (2 * R)) @ (dil(x[a] - x[a + 1]) / (2 * R)) for a in range(0, N, 2)) / N
    assert abs(vT - torch.linalg.eigvalsh(Vd).abs().max().item()) < 1e-10


def test_bounds_B_cover_true_error():
    """Same finite-population check as bound A, for the empirical Bernstein bound."""
    gen = torch.Generator().manual_seed(2)
    m, k, d, n, delta, trials = 6, 3, 2, 200, 0.05, 300
    pop_z = torch.relu(torch.randn(m, k, dtype=DT, generator=gen))
    pop_e = torch.randn(m, d, dtype=DT, generator=gen)
    pop_zb = torch.cat([pop_z, torch.ones(m, 1, dtype=DT)], dim=1)
    M = pop_zb.T @ pop_zb / m
    T = pop_e.T @ pop_zb / m
    fails = 0
    for _ in range(trials):
        idx = torch.randint(0, m, (n,), generator=gen)
        Z, E0, Zb = pop_z[idx], pop_e[idx], pop_zb[idx]
        a_N, b_N = compute_bounds(Z, E0, delta, method="B")
        err_M = torch.linalg.matrix_norm(Zb.T @ Zb / n - M, ord=2).item()
        err_T = (E0.T @ Zb / n - T).norm().item()
        fails += (err_M > a_N) or (err_T > b_N)
    assert fails / trials <= delta


def test_bounds_B_adapt_to_small_variance():
    """Nearly constant features/residuals: B (data variance) must beat A (worst case)."""
    N, k, d = 400, 5, 3
    Z = 1.0 + 0.01 * torch.rand(N, k, dtype=DT)
    E0 = 1.0 + 0.01 * torch.rand(N, d, dtype=DT)
    a_A, b_A = compute_bounds(Z, E0, method="A")
    a_B, b_B = compute_bounds(Z, E0, method="B")
    assert a_B < a_A and b_B < b_A


def test_bounds_B_scaling():
    Z = torch.relu(torch.randn(30, 5, dtype=DT))
    E0 = torch.randn(30, 4, dtype=DT)
    a1, b1 = compute_bounds(Z, E0, method="B")
    a2, b2 = compute_bounds(Z, 3 * E0, method="B")
    assert abs(a2 - a1) < 1e-12 and abs(b2 - 3 * b1) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")

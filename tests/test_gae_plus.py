"""Checks for gae_plus.py. Run: python tests/test_gae_plus.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import math

from gae_plus import (apply_eta, bernstein_radius, chebyshev_radius, compute_bounds, eb_radius,
                      elementwise_bounds, eta_objective_terms, gae_plus_select, paired_variance_norms,
                      select_eta, select_eta_elementwise, split_bounds)

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


def test_chebyshev_bounds_match_image_formula():
    N, k, d, delta = 40, 6, 3, 0.05
    Z = torch.relu(torch.randn(N, k, dtype=DT))
    E0 = torch.randn(N, d, dtype=DT)
    a_N, b_N = compute_bounds(Z, E0, delta, method="chebyshev")
    B = ((Z ** 2).sum(dim=1) + 1).max().item()
    E = E0.norm(dim=1).max().item()
    assert abs(a_N - B * math.sqrt(2 / (N * delta))) < 1e-12              # Z_max^2 sqrt(2/(N delta))
    assert abs(b_N - E * math.sqrt(B) * math.sqrt(2 / (N * delta))) < 1e-12  # E_max Z_max sqrt(2/(N delta))
    assert abs(chebyshev_radius(3.0, 10, 0.25) - 3.0 / math.sqrt(2.5)) < 1e-12


def test_best_is_the_smaller_valid_radius():
    for N in (8, 40, 2000):
        Z = torch.relu(torch.randn(N, 5, dtype=DT))
        E0 = torch.randn(N, 4, dtype=DT)
        a_c, b_c = compute_bounds(Z, E0, method="chebyshev")
        a_A, _ = compute_bounds(Z, E0, method="A")
        a_b, b_b = compute_bounds(Z, E0, method="best")
        B = ((Z ** 2).sum(dim=1) + 1).max().item()
        E = E0.norm(dim=1).max().item()
        b_pin = bernstein_radius(2 * E * math.sqrt(B) / N, E * E * B / N, 2, 0.025)
        assert abs(a_b - min(a_c, a_A)) < 1e-12 and abs(b_b - min(b_c, b_pin)) < 1e-12


def _explicit_split_ranges(Z, Y, A0, A1, A2):
    """Independent reference: Q_i[j,l] = <D_j zb_i, D_l zb_i>, q_i[j] = <e0_i, D_j zb_i>."""
    Zb = torch.cat([Z, torch.ones(len(Z), 1, dtype=DT)], dim=1)
    D1, D2 = A1 - A0, A2 - A1
    L_H, L_g = 0.0, 0.0
    for zb, y in zip(Zb, Y):
        p = torch.stack([D1 @ zb, D2 @ zb])               # [2, d]
        e0 = y - A0 @ zb
        L_H = max(L_H, torch.trace(p @ p.T).item())
        L_g = max(L_g, (p @ e0).norm().item())
    return L_H, L_g


def test_split_bounds_match_explicit_formulas():
    Z, Y, A0, A1, A2 = _random_problem(N=30)
    n, delta = 30, 0.05
    L_H, L_g = _explicit_split_ranges(Z, Y, A0, A1, A2)
    a_c, b_c = split_bounds(Z, Y, A0, A1, A2, delta, method="chebyshev")
    assert abs(a_c - L_H * math.sqrt(2 / (n * delta))) < 1e-9
    assert abs(b_c - L_g * math.sqrt(2 / (n * delta))) < 1e-9
    a_b, b_b = split_bounds(Z, Y, A0, A1, A2, delta, method="best")
    a_t = bernstein_radius(2 * L_H / n, L_H ** 2 / n, 4, delta / 2)
    b_p = bernstein_radius(2 * L_g / n, L_g ** 2 / n, 2, delta / 2)
    assert abs(a_b - min(a_c, a_t)) < 1e-9 and abs(b_b - min(b_c, b_p)) < 1e-9


def test_split_bounds_cover_true_error():
    """Finite population with fixed candidates: exact H, g; bounds must fail in <= delta of draws."""
    gen = torch.Generator().manual_seed(3)
    m, k, d, n, delta, trials = 6, 3, 2, 200, 0.05, 300
    pop_z = torch.relu(torch.randn(m, k, dtype=DT, generator=gen))
    pop_y = torch.randn(m, d, dtype=DT, generator=gen)
    A0, A1, A2 = (torch.randn(d, k + 1, dtype=DT, generator=gen) for _ in range(3))
    H, g, _, _ = eta_objective_terms(pop_z, pop_y, A0, A1, A2)   # uniform weights = exact population
    for method in ("chebyshev", "best"):
        fails = 0
        for _ in range(trials):
            idx = torch.randint(0, m, (n,), generator=gen)
            Hs, gs, _, _ = eta_objective_terms(pop_z[idx], pop_y[idx], A0, A1, A2)
            a_N, b_N = split_bounds(pop_z[idx], pop_y[idx], A0, A1, A2, delta, method=method)
            fails += (torch.linalg.matrix_norm(Hs - H, ord=2).item() > a_N) or ((gs - g).norm().item() > b_N)
        assert fails / trials <= delta, (method, fails)


def test_split_penalty_protects_true_objective():
    """With K = I, J_hat + penalty must upper-bound the true objective whenever the bounds hold."""
    Z, Y, A0, A1, A2 = _random_problem(N=200)
    H, g, _, _ = eta_objective_terms(Z, Y, A0, A1, A2)
    Hs, gs, _, _ = eta_objective_terms(Z[:100], Y[:100], A0, A1, A2)
    a_N = torch.linalg.matrix_norm(Hs - H, ord=2).item()          # exact deviations as "bounds"
    b_N = (gs - g).norm().item()
    for e1, e2 in _feasible_points():
        x = torch.tensor([e1, e2], dtype=DT)
        true_J = (x @ H @ x - 2 * g @ x).item()
        upper = (x @ Hs @ x - 2 * gs @ x).item() + a_N * (x @ x).item() + 2 * b_N * x.norm().item()
        assert true_J <= upper + 1e-9


def test_tied_mode_equals_one_dimensional_search():
    """A1 := A0 and K = diag(0, 1): the selected eta has eta1 = eta2 = t and matches a 1-D search over t."""
    Z, Y, A0, _, A2 = _random_problem(N=60)
    H, g, _, _ = eta_objective_terms(Z, Y, A0, A0, A2)
    K = torch.diag(torch.tensor([0.0, 1.0], dtype=DT))
    for a_N, b_N in ((0.0, 0.0), (0.5, 0.3), (5.0, 2.0)):
        e1, e2, J = select_eta(H, g, K, a_N, b_N)
        assert abs(e1 - e2) < 1e-12
        D = A2 - A0
        best = min(
            (_recon_err(Z, Y, A0 + (i / 100) * D) - _recon_err(Z, Y, A0) + a_N * (i / 100) ** 2 + 2 * b_N * (i / 100), i / 100)
            for i in range(101)
        )
        assert abs(J - best[0]) < 1e-9 and abs(e2 - best[1]) < 1e-9


def _explicit_entries(Z, Y, A0, A1, A2, basis):
    """Independent reference: per-sample (Q11, Q12, Q21, Q22, q1, q2) in the given basis."""
    Zb = torch.cat([Z, torch.ones(len(Z), 1, dtype=DT)], dim=1)
    U1, U2 = A1 - A0, (A2 - A1) if basis == "step" else (A2 - A0)
    rows = []
    for zb, y in zip(Zb, Y):
        p1, p2, e0 = U1 @ zb, U2 @ zb, y - A0 @ zb
        rows.append([p1 @ p1, p1 @ p2, p2 @ p1, p2 @ p2, e0 @ p1, e0 @ p2])
    return torch.tensor(rows, dtype=DT)


def test_elementwise_bounds_match_explicit():
    Z, Y, A0, A1, A2 = _random_problem(N=40)
    n, delta = 40, 0.05
    for basis in ("step", "gae"):
        V = _explicit_entries(Z, Y, A0, A1, A2, basis)
        H, g, rH, rg = elementwise_bounds(Z, Y, A0, A1, A2, delta, "chebyshev", basis)
        assert torch.allclose(H.flatten(), V[:, :4].mean(0)) and torch.allclose(g, V[:, 4:].mean(0))
        s = V.abs().amax(0)
        assert torch.allclose(torch.cat([rH.flatten(), rg]), s / math.sqrt(n * delta / 5))
        _, _, rHb, rgb = elementwise_bounds(Z, Y, A0, A1, A2, delta, "best", basis)
        bern = torch.tensor([bernstein_radius(2 * x / n, x * x / n, 2, delta / 5) for x in s.tolist()], dtype=DT)
        assert torch.allclose(torch.cat([rHb.flatten(), rgb]), torch.minimum(s / math.sqrt(n * delta / 5), bern))


def test_elementwise_bounds_cover_true_error():
    gen = torch.Generator().manual_seed(4)
    m, k, d, n, delta, trials = 6, 3, 2, 200, 0.05, 300
    pop_z = torch.relu(torch.randn(m, k, dtype=DT, generator=gen))
    pop_y = torch.randn(m, d, dtype=DT, generator=gen)
    A0, A1, A2 = (torch.randn(d, k + 1, dtype=DT, generator=gen) for _ in range(3))
    for basis in ("step", "gae"):
        Hp, gp, _, _ = elementwise_bounds(pop_z, pop_y, A0, A1, A2, delta, "best", basis)  # exact means
        for method in ("chebyshev", "best"):
            fails = 0
            for _ in range(trials):
                idx = torch.randint(0, m, (n,), generator=gen)
                H, g, rH, rg = elementwise_bounds(pop_z[idx], pop_y[idx], A0, A1, A2, delta, method, basis)
                fails += bool(((H - Hp).abs() > rH).any() or ((g - gp).abs() > rg).any())
            assert fails / trials <= delta, (basis, method, fails)


def test_elementwise_penalty_protects_true_objective_and_bases_agree():
    Z, Y, A0, A1, A2 = _random_problem(N=200)
    Js = {}
    for basis in ("step", "gae"):
        Hp, gp, _, _ = elementwise_bounds(Z, Y, A0, A1, A2, basis=basis)
        H, g, _, _ = elementwise_bounds(Z[:100], Y[:100], A0, A1, A2, basis=basis)
        rH, rg = (H - Hp).abs(), (g - gp).abs()           # exact deviations as radii
        for e1, e2 in _feasible_points():
            y = torch.tensor([e1, e2] if basis == "step" else [e1 - e2, e2], dtype=DT)
            true_J = (y @ Hp @ y - 2 * gp @ y).item()
            upper = (y @ H @ y - 2 * g @ y + y.abs() @ rH @ y.abs() + 2 * y.abs() @ rg).item()
            assert true_J <= upper + 1e-9
            direct = _recon_err(Z, Y, apply_eta(A0, A1, A2, e1, e2)) - _recon_err(Z, Y, A0)
            assert abs(true_J - direct) < 1e-9             # both bases give the same objective
        Js[basis] = select_eta_elementwise(Hp, gp, torch.zeros(2, 2, dtype=DT), torch.zeros(2, dtype=DT), basis)
    e1, e2, J = select_eta(*eta_objective_terms(Z, Y, A0, A1, A2)[:3])
    for basis in ("step", "gae"):
        assert abs(Js[basis][2] - J) < 1e-9 and Js[basis][:2] == (e1, e2)   # zero radii = no-penalty choice


def test_gae_plus_select_modes_and_defaults():
    Z, Y, A0, A1, A2 = _random_problem(N=80)
    # default = elementwise (A1 - A0, A2 - A0) basis with the best bound
    e1, e2, Ap = gae_plus_select(Z, Y, A0, A1, A2)
    H, g, rH, rg = elementwise_bounds(Z, Y, A0, A1, A2, 0.05, "best", "gae")
    assert (e1, e2) == select_eta_elementwise(H, g, rH, rg, "gae")[:2]
    assert torch.allclose(Ap, apply_eta(A0, A1, A2, e1, e2))
    # tie: eta1 = eta2 = t, A_plus = A0 + t (A2 - A0), 2x2 bounds with K = diag(0, 1)
    t1, t2, At = gae_plus_select(Z, Y, A0, A1, A2, mode="tie")
    Ht, gt, _, _ = eta_objective_terms(Z, Y, A0, A0, A2)
    a_N, b_N = split_bounds(Z, Y, A0, A0, A2, 0.05, "best")
    K = torch.diag(torch.tensor([0.0, 1.0], dtype=DT))
    assert t1 == t2 and (t1, t2) == select_eta(Ht, gt, K, a_N, b_N)[:2]
    assert torch.allclose(At, apply_eta(A0, A1, A2, t1, t2))     # same A_plus in the original eta
    # matrix: free eta, 2x2 operator-norm bound, K = I
    m1, m2, Am = gae_plus_select(Z, Y, A0, A1, A2, mode="matrix")
    Hm, gm, _, _ = eta_objective_terms(Z, Y, A0, A1, A2)
    a_m, b_m = split_bounds(Z, Y, A0, A1, A2, 0.05, "best")
    assert (m1, m2) == select_eta(Hm, gm, torch.eye(2, dtype=DT), a_m, b_m)[:2]
    assert torch.allclose(Am, apply_eta(A0, A1, A2, m1, m2))
    # chebyshev is selectable; unknown options are rejected
    gae_plus_select(Z, Y, A0, A1, A2, bound="chebyshev")
    for kw in ({"mode": "free"}, {"bound": "A"}):
        try:
            gae_plus_select(Z, Y, A0, A1, A2, **kw)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {kw}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")

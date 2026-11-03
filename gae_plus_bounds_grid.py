"""GAE+ bounds grid: a_N/b_N by Chebyshev or best inequality, sample split off/on.

OOD activations are shuffled once and split into fit (n_fit) and test (n_test).
Split off: candidates and H, g, K, a_N, b_N all from the fit set.
Split on : candidates from the first half of the fit set, H, g, a_N, b_N from the
           second half, K = I. The selected A_plus is scored on the untouched test set.
"""
import argparse
import json
import os

import torch

from config import Config
from gae_plus import (apply_eta, compute_bounds, elementwise_bounds, eta_objective_terms, select_eta,
                      select_eta_elementwise, split_bounds)
from gae_plus_probe import F64, build_candidates, encode, load_activations
from gae_plus_split_check import recon_err
from utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ood_set", default="halu_eval")
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--split", choices=["off", "on"], required=True)
    p.add_argument("--n_fit", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    set_seed(Config.SEED)

    model, layer, explainer, ood_in, ood_out, id_in, id_out = load_activations(
        args.model, args.checkpoint, args.ood_set)
    assert len(ood_in) >= args.n_fit + args.n_test, (len(ood_in), args.n_fit, args.n_test)
    perm = torch.randperm(len(ood_in), generator=torch.Generator().manual_seed(Config.SEED))
    fit, test = perm[:args.n_fit], perm[args.n_fit:args.n_fit + args.n_test]
    if args.split == "off":
        build_idx, sel_idx = fit, fit
    else:
        build_idx, sel_idx = fit[:args.n_fit // 2], fit[args.n_fit // 2:]

    A0, A1, A2, n_build = build_candidates(
        explainer, model, layer, id_in, id_out, ood_in[build_idx], ood_out[build_idx], args.r)
    Z_all, _ = encode(explainer, ood_in)
    Y_all = ood_out.to(**F64)
    Zs, Ys = Z_all[sel_idx], Y_all[sel_idx]
    Zt, Yt = Z_all[test], Y_all[test]

    E0s = Ys - Zs @ A0[:, :-1].T - A0[:, -1]

    res = {
        "model": args.model, "ood_set": args.ood_set, "r": args.r, "delta": args.delta,
        "split": args.split, "n_build": n_build, "n_select": len(sel_idx), "n_test": len(test),
        "subspace_matrix": Config.GAE_SUBSPACE_MATRIX, "step1_match_norms": Config.GAE_STEP1_MATCH_NORMS,
        "lam_geom": Config.GAE_AC_LAMBDA_GEOM, "lam_pres(lam_gae)": Config.GAE_AC_LAMBDA_GAE,
        "lam_id": Config.GAE_AC_LAMBDA_ID,
        "test": {"R(A0)": recon_err(Zt, Yt, A0), "R(A1)": recon_err(Zt, Yt, A1), "R(A2)": recon_err(Zt, Yt, A2)},
    }
    # split off: original plan, K = <D_j, D_l>_F; split on: mode "matrix", K = I. Both in the step basis.
    H, g, K_D, R0_sel = eta_objective_terms(Zs, Ys, A0, A1, A2, basis="step")
    K = K_D if args.split == "off" else torch.eye(2, **F64)
    res["matrix"] = {"H": H.tolist(), "g": g.tolist(), "K": K.tolist(), "R0_select": R0_sel}
    for method in ("none", "chebyshev", "best"):
        if method == "none":
            a_N, b_N = 0.0, 0.0
        elif args.split == "off":
            a_N, b_N = compute_bounds(Zs, E0s, args.delta, method=method)
        else:
            a_N, b_N = split_bounds(Zs, Ys, A0, A1, A2, args.delta, method=method, basis="step")
        e1, e2, J = select_eta(H, g, K, a_N, b_N, basis="step")
        res["matrix"][method] = {"a_N": a_N, "b_N": b_N, "eta1": e1, "eta2": e2, "J": J,
                                 "R_test(A_plus)": recon_err(Zt, Yt, apply_eta(A0, A1, A2, e1, e2))}
        print(f"[split={args.split} matrix] {method}: a_N={a_N:.4g} b_N={b_N:.4g} -> eta=({e1:.2f}, {e2:.2f}) "
              f"R_test={res['matrix'][method]['R_test(A_plus)']:.4g} "
              f"(A0 {res['test']['R(A0)']:.4g}, A2 {res['test']['R(A2)']:.4g})")

    # element-wise (Bonferroni over H11, H12, H22, g1, g2): needs held-out samples -> split on
    if args.split == "on":
        res["elementwise"] = {}
        for basis in ("step", "gae"):
            for method in ("chebyshev", "best"):
                H, g, rH, rg = elementwise_bounds(Zs, Ys, A0, A1, A2, args.delta, method, basis)
                e1, e2, J = select_eta_elementwise(H, g, rH, rg, basis)
                key = f"{basis}_{method}"
                res["elementwise"][key] = {"H": H.tolist(), "g": g.tolist(), "rH": rH.tolist(), "rg": rg.tolist(),
                                           "eta1": e1, "eta2": e2, "J": J,
                                           "R_test(A_plus)": recon_err(Zt, Yt, apply_eta(A0, A1, A2, e1, e2))}
                print(f"[split=on elementwise {basis}] {method}: rH={[[round(x, 3) for x in row] for row in rH.tolist()]} "
                      f"rg={[round(x, 3) for x in rg.tolist()]} -> eta=({e1:.2f}, {e2:.2f}) "
                      f"R_test={res['elementwise'][key]['R_test(A_plus)']:.4g}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"bounds_grid_split_{args.split}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

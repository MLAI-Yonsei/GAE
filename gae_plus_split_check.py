"""Design check for sample splitting (not a result).

1) How many features never fire in the N OOD samples (and fire only in the other half)?
2) Build A1/A2 on a random half A, then compare reconstruction error on A (fit)
   vs the other half B (held out), and the no-penalty eta chosen on each.
"""
import argparse
import json
import os

import torch

from config import Config
from gae_plus import eta_objective_terms, select_eta
from gae_plus_probe import F64, build_candidates, encode, load_activations
from utils import set_seed


def recon_err(Z, Y, A):
    """mean_i ||y_i - W z_i - b||^2 with A = [W, b]."""
    return ((Y - Z @ A[:, :-1].T - A[:, -1]) ** 2).sum(1).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ood_set", default="halu_eval")
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    set_seed(Config.SEED)

    model, layer, explainer, ood_in, ood_out, id_in, id_out = load_activations(
        args.model, args.checkpoint, args.ood_set)
    N = len(ood_in)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(Config.SEED))
    a, b = perm[: N // 2], perm[N // 2:]          # half A builds the candidates, half B is held out

    Z_all, _ = encode(explainer, ood_in)
    active = Z_all > 0
    res = {
        "model": args.model, "ood_set": args.ood_set, "r": args.r, "N": N, "k": Z_all.shape[1],
        "mean_active_features_per_sample": active.sum(1).double().mean().item(),
        "never_active_all_N": int((~active.any(0)).sum()),
        "never_active_in_A": int((~active[a].any(0)).sum()),
        "inactive_in_A_but_active_in_B": int((~active[a].any(0) & active[b].any(0)).sum()),
    }

    A0, A1, A2, n_fit = build_candidates(explainer, model, layer, id_in, id_out, ood_in[a], ood_out[a], args.r)
    for name, idx in (("half_A_fit", a[:n_fit]), ("half_B_heldout", b)):
        Z, Y = Z_all[idx], ood_out[idx].to(**F64)
        H, g, K, R0 = eta_objective_terms(Z, Y, A0, A1, A2)
        e1, e2, J = select_eta(H, g, K)
        res[name] = {
            "n": len(idx), "R(A0)": R0, "R(A1)": recon_err(Z, Y, A1), "R(A2)": recon_err(Z, Y, A2),
            "eta_no_penalty": [e1, e2], "J": J, "H": H.tolist(), "g": g.tolist(),
        }
        print(name, json.dumps(res[name]))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "split_check.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

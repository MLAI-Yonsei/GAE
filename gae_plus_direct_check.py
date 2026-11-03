"""Check: A0, A1, A2 built with gae.py directly (GAE_PLUS_README §1) match build_candidates
(run_experiment.evaluate_baseline_gae path) on the same OOD samples, and give the same eta."""
import argparse
import json
import os

import torch

from config import Config
from gae import fit_affine_constrained_decoder, gae_torch
from gae_plus import gae_plus_select
from gae_plus_probe import F64, build_candidates, encode, load_activations
from utils import set_seed


def direct_candidates(explainer, H_in, H_out, r, lam_geom, lam_id, lam_gae, fit_samples, match_norms):
    """README §1: Step 1 (gae_torch) + optional column-norm match + Step 2 (fit_affine_constrained_decoder)."""
    sae = explainer.saelens
    B_id, b0 = sae.W_dec.T.detach().clone(), sae.b_dec_out.detach().clone()   # transcoder: output bias
    B1, geo = gae_torch(B_id, H_out, r=r, subspace_matrix=Config.GAE_SUBSPACE_MATRIX)
    if match_norms:
        B1 = B1 * (B_id.norm(dim=0) / (B1.norm(dim=0) + 1e-8))
    W2, b2, _ = fit_affine_constrained_decoder(
        explainer=explainer, activations_in=H_in, activations_target=H_out,
        B_original=B_id, B_gae=B1, U_ood=geo["U_ood"],
        lam_geom=lam_geom, lam_id=lam_id, lam_gae=lam_gae, fit_samples=fit_samples)
    A0 = torch.cat([B_id, b0[:, None]], 1).to(**F64)
    A1 = torch.cat([B1, b0[:, None]], 1).to(**F64)
    A2 = torch.cat([W2, b2[:, None]], 1).to(**F64)
    return A0, A1, A2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ood_set", default="halu_eval")
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--n_build", type=int, default=1000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    set_seed(Config.SEED)

    model, layer, explainer, ood_in, ood_out, id_in, id_out = load_activations(
        args.model, args.checkpoint, args.ood_set)
    a, b = slice(0, args.n_build), slice(args.n_build, 2 * args.n_build)
    ref = build_candidates(explainer, model, layer, id_in, id_out, ood_in[a], ood_out[a], args.r)[:3]
    H_in, H_out = ood_in[a].to(Config.device), ood_out[a].to(Config.device)
    new = direct_candidates(explainer, H_in, H_out, args.r, Config.GAE_AC_LAMBDA_GEOM, Config.GAE_AC_LAMBDA_ID,
                            Config.GAE_AC_LAMBDA_GAE, Config.GAE_AC_FIT_SAMPLES, Config.GAE_STEP1_MATCH_NORMS)

    Z_ref, _ = encode(explainer, ood_in[b])
    with torch.no_grad():
        _, Z_new = explainer(ood_in[b].to(Config.device), return_z=True)   # README: explainer(..., return_z=True)
    Z_new = Z_new.to(**F64)
    Y = ood_out[b].to(**F64)

    res = {"n_build": args.n_build, "n_select": len(Y), "r": args.r,
           "lam_geom": Config.GAE_AC_LAMBDA_GEOM, "lam_id": Config.GAE_AC_LAMBDA_ID,
           "lam_gae": Config.GAE_AC_LAMBDA_GAE, "fit_samples": Config.GAE_AC_FIT_SAMPLES,
           "subspace_matrix": Config.GAE_SUBSPACE_MATRIX, "step1_match_norms": Config.GAE_STEP1_MATCH_NORMS}
    for name, x, y in (("A0", ref[0], new[0]), ("A1", ref[1], new[1]), ("A2", ref[2], new[2]), ("Z", Z_ref, Z_new)):
        res[name] = {"max_abs_diff": (x - y).abs().max().item(),
                     "rel_fro_diff": ((x - y).norm() / x.norm()).item()}
    for mode in ("elementwise", "matrix", "nosplit"):
        e_ref = gae_plus_select(Z_ref, Y, *ref, mode=mode)[:2]
        e_new = gae_plus_select(Z_new, Y, *new, mode=mode)[:2]
        res[f"eta_{mode}"] = {"build_candidates": e_ref, "gae_py": e_new}
    print(json.dumps(res, indent=2))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "direct_check.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

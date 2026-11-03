"""First GAE+ probe: do bounds A/B leave a non-zero (eta1, eta2)?

Builds A0/A1/A2 with the same code path as run_experiment.py (adv mode, GAE
decoder_mode="affine"), then reports a_N, b_N and the selected (eta1, eta2)
for three cases: no penalty, bound A, bound B. Ranges are observed maxima
(no guarantee); see the GAE+ decision log.
"""
import argparse
import copy
import json
import os

import torch

from config import Config
from gae_plus import compute_bounds, eta_objective_terms, paired_variance_norms, select_eta
from ood_utils import data_adv
from ood_utils.data_adv import load_ood_dataset, sample_and_tokenize_dataset_by_token_budget
from run_experiment import evaluate_baseline_gae, load_explainer
from train_explainers import collect_id_activations
from utils import collect_mlp_in_out_activations_batch, get_layer, load_model, set_seed


F64 = dict(device=Config.device, dtype=torch.float64)


def load_activations(model_name, checkpoint, ood_set, ood_token_budget=2_000_000):
    """Model, explainer, OOD (N_COV, last position) and ID activations, as in run_experiment.py adv."""
    dev = Config.device
    model, tokenizer = load_model(model_name)
    layer = get_layer(model_name)
    explainer = load_explainer(checkpoint, "transcoder", model.cfg.d_model, model.cfg.d_vocab)
    ds, text_field = load_ood_dataset(ood_set, split="train")
    if not hasattr(ds, "__getitem__"):
        from datasets import Dataset
        ds = Dataset.from_list(list(ds))
    min_len = min(Config.MIN_LEN, 4) if ood_set == "halu_eval" else Config.MIN_LEN
    tokens, _ = sample_and_tokenize_dataset_by_token_budget(
        ds, text_field, tokenizer, token_budget=ood_token_budget, max_length=Config.MAX_LEN,
        min_length=min_len, seed=Config.SEED, max_samples=Config.N_OOD_DOMAIN)
    ood_in, ood_out = collect_mlp_in_out_activations_batch(
        model, tokens, layer, position=-1,
        hook_in=getattr(explainer, "input_hook_point", "ln2.hook_normalized"),
        hook_out=getattr(explainer, "output_hook_point", "hook_mlp_out"), device=dev)
    id_in, id_out, _ = collect_id_activations(
        model, tokenizer, Config.N_COV, layer, use_cache=False, data_module=data_adv)
    return model, layer, explainer, ood_in[:Config.N_COV], ood_out[:Config.N_COV], id_in, id_out[:Config.N_COV]


def build_candidates(explainer, model, layer, id_in, id_out, ood_in, ood_out, r):
    """A0, A1, A2 ([d, k+1], float64) from GAE affine fit on the given OOD samples."""
    sae = explainer.saelens
    W0, b0 = sae.W_dec.T.detach().clone(), sae.b_dec_out.detach().clone()
    gae = evaluate_baseline_gae(
        copy.deepcopy(explainer), model, id_in, ood_in, layer, "transcoder", r=r,
        id_activations_out=id_out, ood_activations_out=ood_out, decoder_mode="affine")
    A0 = torch.cat([W0, b0[:, None]], 1).to(**F64)
    A1 = torch.cat([gae.B_gae.to(b0.device), b0[:, None]], 1).to(**F64)
    A2 = torch.cat([gae.gae_decoder_override, gae.gae_decoder_bias[:, None]], 1).to(**F64)
    return A0, A1, A2, gae.gae_extra_metadata["affine_constrained"]["fit_samples"]


def encode(explainer, X):
    """Frozen-encoder features z and the explainer's own reconstruction, float64."""
    with torch.no_grad():
        outs = [explainer(X[i:i + 128].to(Config.device), return_z=True) for i in range(0, len(X), 128)]
    return torch.cat([o[1] for o in outs]).to(**F64), torch.cat([o[0] for o in outs]).to(**F64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ood_set", default="halu_eval")
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--ood_token_budget", type=int, default=2_000_000)  # run_experiment default
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    set_seed(Config.SEED)
    dev = Config.device
    f64 = F64

    model, layer, explainer, ood_in, ood_out, id_in, id_out = load_activations(
        args.model, args.checkpoint, args.ood_set, args.ood_token_budget)
    A0, A1, A2, n_fit = build_candidates(explainer, model, layer, id_in, id_out, ood_in, ood_out, args.r)

    # same N samples as the Step 2 fit (first n_fit), z from the frozen encoder
    Z, recon = encode(explainer, ood_in[:n_fit])
    Y = ood_out[:n_fit].to(**f64)
    E0 = Y - Z @ A0[:, :-1].T - A0[:, -1]
    recon_rel_gap = ((recon - (Y - E0)).norm() / recon.norm()).item()  # A0 z_bar vs explainer output

    H, g, K, R0 = eta_objective_terms(Z, Y, A0, A1, A2)
    perm = torch.randperm(n_fit, generator=torch.Generator().manual_seed(Config.SEED)).to(dev)
    Zp, Ep = Z[perm], E0[perm]                     # shuffled order for the pairs of bound B
    B = ((Zp ** 2).sum(1) + 1).max().item()
    E = Ep.norm(dim=1).max().item()
    vM, vT = paired_variance_norms(Zp, Ep, B, E * B ** 0.5)

    def change(e1, e2):                            # sample error change x^T H x - 2 g^T x
        x = torch.tensor([e1, e2], **f64)
        return (x @ H @ x - 2 * g @ x).item()

    res = {
        "model": args.model, "ood_set": args.ood_set, "r": args.r, "delta": args.delta,
        "N": n_fit, "k": Z.shape[1], "d": Y.shape[1],
        "subspace_matrix": Config.GAE_SUBSPACE_MATRIX, "step1_match_norms": Config.GAE_STEP1_MATCH_NORMS,
        "lam_geom": Config.GAE_AC_LAMBDA_GEOM, "lam_pres(lam_gae)": Config.GAE_AC_LAMBDA_GAE,
        "lam_id": Config.GAE_AC_LAMBDA_ID,
        "recon_rel_gap_A0_vs_explainer": recon_rel_gap,
        "R0": R0, "R0_per_dim": R0 / Y.shape[1],
        "H": H.tolist(), "g": g.tolist(), "K": K.tolist(),
        "ranges_observed": {"B=max||z_bar||^2": B, "E_max=max||e0||": E},
        "paired_variance": {"vM": vM, "vT": vT},
        "sample_change_at": {"(1,0) Step 1": change(1, 0), "(0,1) GAE": change(0, 1)},
    }
    for name in ("none", "A", "B"):
        a_N, b_N = (0.0, 0.0) if name == "none" else compute_bounds(Zp, Ep, args.delta, method=name)
        e1, e2, J = select_eta(H, g, K, a_N, b_N)
        res[name] = {"a_N": a_N, "b_N": b_N, "eta1": e1, "eta2": e2, "J": J,
                     "sample_change": change(e1, e2), "penalty": J - change(e1, e2)}
        print(f"[{name}] a_N={a_N:.4g} b_N={b_N:.4g} -> eta=({e1:.2f}, {e2:.2f}) J={J:.4g}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

"""
GAE+ : choose how much of GAE's two updates to apply, using only the adaptation sample S.

Decoders (affine, A = [W, b]):  A0 = Fixed,  A1 = Step 1 (aligned dictionary, original bias),  A2 = Step 2 (refit).
Candidates:  A(eta) = (1 - eta1 - eta2) A0 + eta1 A1 + eta2 A2  for eta in ETA_GRID.
Each candidate is scored on S with the repository's deletion scores (compute_feature_mask_scores) and compared with
Fixed through a utility score:
  * fallback : on inputs where the candidate's full-vs-empty margin is <= tau, Fixed's scores are used instead;
  * common denominator : deletion effects are divided by Fixed's margin, on inputs where Fixed's margin is > tau;
  * clip : the nAOPC-like score clips each input to [-1, 1]; the nComp-like score does not;
  * utility = min over the two scores of (q - q_fixed) / max(|q_fixed|, 0.1).
The candidate with the largest utility is kept (ties -> the smaller change). No bootstrap, no staged search.

GAE itself (gae_torch, fit_affine_constrained_decoder) and the evaluation scores come from the main code; the only
additions are the Step 1 covariance shrinkage, the fixed-bias refit and the automatic Step 2 ridge (all optional).
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from gae import extract_dictionary_from_explainer, fit_affine_constrained_decoder, gae_torch
from ood_utils.evaluation import compute_aopc_and_normalized, compute_comp_suff, compute_feature_mask_scores
from ood_utils.saelens_adapters import SaeLensSparseAutoencoderAdapter, SaeLensTranscoderAdapter
from sae_training.sparse_autoencoder import SparseAutoencoder as SaeLensSparseAutoencoder
from utils import (
    collect_hook_activations_batch,
    collect_mlp_in_out_activations_batch,
    extract_hook_activations,
    get_layer,
    get_true_logits_from_model,
    load_model,
    set_seed,
)

ETA_GRID = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.0)]
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
TAU = Config.FAITH_NORM_MIN_DELTA  # margin threshold shared by the fallback and the valid-input rule


def mix(A0, A1, A2, eta):
    """A(eta) for decoders given as (W, b) pairs."""
    e1, e2 = eta
    return tuple((1 - e1 - e2) * a0 + e1 * a1 + e2 * a2 for a0, a1, a2 in zip(A0, A1, A2))


def gate(rows, fixed_rows, tau=TAU):
    """Input-level fallback: keep the candidate's row only where its full-vs-empty margin exceeds tau."""
    return [r if r["s0"] - r["s_empty"] > tau else f for r, f in zip(rows, fixed_rows)]


def surrogate(rows, fixed_rows, tau=TAU):
    """(clipped mean, unclipped mean) of the gated deletion effect divided by Fixed's margin, on Fixed-valid inputs."""
    x = np.array([
        (r["s0"] - np.mean(r["s_del"])) / (f["s0"] - f["s_empty"])
        for r, f in zip(gate(rows, fixed_rows, tau), fixed_rows)
        if f["s0"] - f["s_empty"] > tau
    ])
    return (float(np.clip(x, -1.0, 1.0).mean()), float(x.mean())) if x.size else (float("nan"), float("nan"))


def utility(rows, fixed_rows, tau=TAU):
    """Worst relative gain over Fixed across the clipped and unclipped scores (0 for Fixed itself)."""
    q, qf = surrogate(rows, fixed_rows, tau), surrogate(fixed_rows, fixed_rows, tau)
    return min((q[i] - qf[i]) / max(abs(qf[i]), 0.1) for i in (0, 1))


def select(candidates, fixed_rows, tau=TAU):
    """candidates: list of dicts with 'eta' and 'rows'. Adds 'utility' and returns the winner."""
    for c in candidates:
        c["utility"] = 0.0 if c["rows"] is fixed_rows else utility(c["rows"], fixed_rows, tau)
    ok = [c for c in candidates if np.isfinite(c["utility"])]
    return max(ok, key=lambda c: (c["utility"], -sum(c["eta"]), -c["eta"][1])) if ok else candidates[0]


def aggregate(rows, tau=TAU):
    """nAOPC / nComp as in the main evaluation: per-input scores from the main functions, mean over valid inputs."""
    a = [compute_aopc_and_normalized(dict(r, s_rand=[]), M_VALUES) for r in rows]
    c = [compute_comp_suff(dict(r, s_keep=[]), M_VALUES) for r in rows]
    valid = [i for i, x in enumerate(a) if x["delta_max"] > tau]
    if not valid:
        return dict(n_aopc=None, n_comp=None, n_valid=0, n_eval=len(rows))
    return dict(
        n_aopc=float(np.mean([a[i]["n_aopc"] for i in valid])),
        n_comp=float(np.mean([c[i]["n_comp"] for i in valid])),
        n_valid=len(valid),
        n_eval=len(rows),
    )


def with_decoder(explainer, W, b):
    """Shallow copy of the explainer that decodes with (W, b); the encoder is shared and untouched."""
    e = copy.copy(explainer)
    e.gae_decoder_override, e.gae_decoder_bias = W, b
    return e


@torch.no_grad()
def score_rows(model, explainer, tokens_list, layer, model_type, desc="scoring"):
    """Per-input deletion scores (s0, s_empty, s_del at M_VALUES) at the last position, target = model's argmax."""
    device = Config.device
    in_hook, out_hook = explainer.input_hook_point, explainer.output_hook_point
    rows = []
    for tokens in tqdm(tokens_list, desc=desc, leave=False):
        t = tokens.unsqueeze(0).to(device)
        h = extract_hook_activations(model, t, layer, position=-1, hook_name=in_hook)
        h_out = None
        if explainer.output_kind == "mlp_out" and out_hook:
            h_out = extract_hook_activations(model, t, layer, position=-1, hook_name=out_hook)
        logits = get_true_logits_from_model(model, t)[:, -1, :]
        s = compute_feature_mask_scores(
            explainer=explainer, model=model, tokens=t, layer=layer, position=-1, h=h, h_out=h_out,
            logits_original=logits, target_token_idx=int(logits[0].argmax()), model_type=model_type,
            m_values=M_VALUES, compute_random_baseline=False, compute_sufficiency=False,
        )
        rows.append(dict(s0=float(s["s0"]), s_empty=float(s["s_empty"]), s_del=[float(v) for v in s["s_del"]]))
    return rows


def load_explainer(path, model_type):
    ckpt = torch.load(path, map_location=Config.device, weights_only=False)
    cfg = ckpt["cfg"]
    cfg.device = str(Config.device)
    sae = SaeLensSparseAutoencoder(cfg)
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(Config.device).eval()
    is_tc = bool(getattr(cfg, "is_transcoder", False))
    if is_tc != (model_type == "transcoder"):
        raise ValueError(f"Checkpoint is_transcoder={is_tc} but --explainer {model_type}")
    return SaeLensTranscoderAdapter(sae) if is_tc else SaeLensSparseAutoencoderAdapter(sae)


def load_tokens(path, tokenizer=None, max_len=128):
    """Records from .jsonl / .json with 'tokens' (or 'context_ids'); 'text' is tokenized with BOS and truncated."""
    text = Path(path).read_text()
    records = json.loads(text) if str(path).endswith(".json") else [json.loads(l) for l in text.splitlines() if l.strip()]
    out = []
    for r in records:
        ids = r.get("tokens", r.get("context_ids"))
        if ids is None:
            ids = ([tokenizer.bos_token_id] + tokenizer.encode(r["text"]))[:max_len]
        out.append(torch.tensor(ids, dtype=torch.long))
    return out


def collect(model, explainer, tokens_list, layer):
    """(decoder inputs X, decoder targets Y) at the last position; Y = X for an SAE."""
    if explainer.output_kind == "mlp_out":
        return collect_mlp_in_out_activations_batch(
            model, tokens_list, layer, position=-1,
            hook_in=explainer.input_hook_point, hook_out=explainer.output_hook_point, device=Config.device,
        )
    X = collect_hook_activations_batch(
        model, tokens_list, layer, position=-1, hook_name=explainer.input_hook_point, device=Config.device
    )
    return X, X


def build_decoders(explainer, model_type, X, Y, args, Y_id=None):
    """A0, A1 and the Step 2 decoder(s) from the main GAE functions. Returns (A0, A1, {bias_mode: A2}, info)."""
    dev = Config.device
    sae = explainer.saelens
    b0 = (sae.b_dec_out if model_type == "transcoder" else sae.b_dec).detach().to(dev)
    B = extract_dictionary_from_explainer(explainer, X.to(dev), model_type, dict_space="auto").to(dev)
    B_gae, info = gae_torch(B, Y.to(dev), args.rank, shrink=None if args.shrink == "none" else args.shrink, H_id=Y_id)
    B_gae = B_gae * (B.norm(dim=0) / (B_gae.norm(dim=0) + 1e-8)).unsqueeze(0)  # match dictionary scale, as in main
    step2, fit_info = {}, {}
    for mode in (["free", "fixed"] if args.bias == "both" else [args.bias]):
        W2, b2, fi = fit_affine_constrained_decoder(
            explainer=explainer, activations_in=X.to(dev), activations_target=Y.to(dev), B_original=B, B_gae=B_gae,
            U_ood=info["U_ood"], lam_geom=args.lam_geom, lam_id=args.lam_id, lam_gae=args.lam_gae,
            fit_samples=len(X), solver_device=str(dev), fixed_bias=b0 if mode == "fixed" else None,
            lam_auto=args.lam_auto,
        )
        step2[mode] = (W2.to(dev), b2.to(dev))
        fit_info[mode] = {k: v for k, v in fi.items() if isinstance(v, (int, float, str, bool))}
    return (B, b0), (B_gae, b0), step2, dict(shrink_rho=info["shrink_rho"], step2=fit_info)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(Config.MODELS))
    p.add_argument("--explainer", required=True, choices=["transcoder", "sae"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adaptation", required=True, help="adaptation records (.jsonl/.json with 'tokens')")
    p.add_argument("--evaluation", required=True, help="evaluation records, disjoint from the adaptation records")
    p.add_argument("--output", required=True)
    p.add_argument("--n-fit", type=int, default=128, help="use the first N adaptation records as S")
    p.add_argument("--n-eval", type=int, default=None)
    # GAE options (same meaning as in main; sum-loss ridge units)
    p.add_argument("--rank", type=int, default=Config.OOD_SUBSPACE_RANK_R)
    p.add_argument("--lam-geom", type=float, default=Config.GAE_AC_LAMBDA_GEOM)
    p.add_argument("--lam-gae", type=float, default=Config.GAE_AC_LAMBDA_GAE)
    p.add_argument("--lam-id", type=float, default=Config.GAE_AC_LAMBDA_ID)
    # additions
    p.add_argument("--shrink", choices=["decoder", "id", "none"], default="decoder", help="Step 1 covariance shrinkage target")
    p.add_argument("--id-data", help="ID records (.jsonl/.json with 'tokens' or 'text'); required for --shrink id")
    p.add_argument("--lam-auto", action="store_true", help="Step 2: replace --lam-gae by the Ledoit-Wolf ridge")
    p.add_argument("--bias", choices=["free", "fixed", "both"], default="free", help="Step 2 bias: refit (main), keep b0, or both families")
    p.add_argument("--gate", choices=["on", "off"], default="on", help="apply the Fixed fallback when reporting evaluation scores")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    set_seed(Config.SEED)
    model, tokenizer = load_model(args.model)
    layer = get_layer(args.model)
    explainer = load_explainer(args.checkpoint, args.explainer)

    S = load_tokens(args.adaptation, tokenizer)[: args.n_fit]
    E = load_tokens(args.evaluation, tokenizer)[: args.n_eval]
    X, Y = collect(model, explainer, S, layer)
    Y_id = None
    if args.shrink == "id":
        if not args.id_data:
            p.error("--shrink id requires --id-data")
        Y_id = collect(model, explainer, load_tokens(args.id_data, tokenizer), layer)[1]

    A0, A1, step2, info = build_decoders(explainer, args.explainer, X, Y, args, Y_id)

    # candidates on the eta grid (the Step 1-only points do not depend on the bias family)
    candidates = [dict(eta=(0.0, 0.0), bias="-", decoder=None)]
    for mode, A2 in step2.items():
        for eta in ETA_GRID[1:]:
            if eta[1] == 0.0 and mode != list(step2)[0]:
                continue
            candidates.append(dict(eta=eta, bias=mode if eta[1] > 0 else "-", decoder=mix(A0, A1, A2, eta)))

    fixed_S = score_rows(model, explainer, S, layer, args.explainer, "S: Fixed")
    for c in candidates:
        c["rows"] = fixed_S if c["decoder"] is None else score_rows(
            model, with_decoder(explainer, *c["decoder"]), S, layer, args.explainer, f"S: eta={c['eta']} {c['bias']}"
        )
    best = select(candidates, fixed_S)
    table = []
    print()  # the progress bar leaves the cursor mid-line
    for c in candidates:
        q = surrogate(c["rows"], fixed_S)
        table.append(dict(eta=c["eta"], bias=c["bias"], utility=c["utility"], q_clipped=q[0], q_unclipped=q[1], selected=c is best))
        print(f"eta={c['eta']} bias={c['bias']:5s} utility={c['utility']:+.4f}  q={q[0]:.4f} / {q[1]:.4f}{'  <- selected' if c is best else ''}")

    # evaluation: Fixed, GAE (eta = (0, 1), first bias family) and the selected candidate
    fixed_E = score_rows(model, explainer, E, layer, args.explainer, "eval: Fixed")
    gae = next(c for c in candidates if c["eta"] == (0.0, 1.0))
    report = {"fixed": dict(aggregate(fixed_E), eta=(0.0, 0.0))}
    rows_E = {"fixed": fixed_E}
    for name, c in (("gae", gae), ("gae_plus", best)):
        if c["decoder"] is None:
            rows = fixed_E
        elif name == "gae_plus" and c is gae:
            rows = rows_E["gae"]
        else:
            rows = score_rows(model, with_decoder(explainer, *c["decoder"]), E, layer, args.explainer, f"eval: {name}")
        rows_E[name] = rows
        gated = gate(rows, fixed_E)
        both = dict(with_gate=aggregate(gated), without_gate=aggregate(rows), fallback=sum(g is f for g, f in zip(gated, fixed_E)) if rows is not fixed_E else 0)
        report[name] = dict(both["with_gate" if args.gate == "on" else "without_gate"], eta=c["eta"], bias=c["bias"], **both)
    print()
    for name, r in report.items():
        print(f"[eval] {name:9s} nAOPC={r['n_aopc']:.4f} nComp={r['n_comp']:.4f} valid={r['n_valid']}/{r['n_eval']} eta={r['eta']}")

    (out / "result.json").write_text(json.dumps(dict(args=vars(args), n_fit=len(S), info=info, selection=table, evaluation=report), indent=2) + "\n")
    with (out / "rows.jsonl").open("w") as f:
        for c in candidates:
            f.write(json.dumps(dict(split="S", eta=c["eta"], bias=c["bias"], rows=c["rows"])) + "\n")
        for name, rows in rows_E.items():
            f.write(json.dumps(dict(split="eval", name=name, rows=rows)) + "\n")


if __name__ == "__main__":
    main()

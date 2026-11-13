"""Pinned public GAE causal metrics and scalar feature-deletion evaluation."""
from __future__ import annotations
import math
import statistics
import numpy as np
import torch
M_VALUES=[1,2,4,8,16,32,64,128]
FAITH_RANK_MODES=('activation','causal_topz')


def version(profile='medical'):
    if profile not in ['medical','standard']:raise ValueError('Unknown metric profile')
    return 'gae_public_c6_medical_v1' if profile=='medical' else 'gae_public_c6_halueval_v1'

def _rank_features_single_example(
    z: torch.Tensor,  # [1, k]
    target_logit_from_features_fn,
    base_score: float,
    rank_mode: str = "causal_topz",
    rank_top_f: int = 64,
) -> torch.Tensor:
    """
    Return ranked feature indices [k] for a single example.
    - activation: rank by feature activation z
    - causal_topz: compute causal effects on top-F activated features, then sort by effect
    """
    if rank_mode not in FAITH_RANK_MODES:
        raise ValueError(f"Unknown rank_mode='{rank_mode}'. Use one of {FAITH_RANK_MODES}.")

    k = z.shape[1]
    feature_scores = z[0]
    act_order = torch.argsort(feature_scores, dim=0, descending=True)  # [k]
    if rank_mode == "activation" or k <= 1:
        return act_order

    F = int(max(1, min(int(rank_top_f), k)))
    cand = act_order[:F]  # top-F by activation
    effects = []
    with torch.no_grad():
        for idx in cand.tolist():
            feats_abl = z.clone()
            feats_abl[0, idx] = 0.0
            score_abl = float(target_logit_from_features_fn(feats_abl))
            effects.append(base_score - score_abl)

    effect_tensor = torch.tensor(effects, device=z.device, dtype=feature_scores.dtype)
    order_eff = torch.argsort(effect_tensor, descending=True)
    ranked_cand = cand[order_eff]
    if F >= k:
        return ranked_cand

    mask = torch.ones(k, dtype=torch.bool, device=z.device)
    mask[ranked_cand] = False
    remaining = act_order[mask[act_order]]
    return torch.cat([ranked_cand, remaining], dim=0)

def compute_aopc_and_normalized(
    scores: Dict[str, object],
    m_values: List[int],
    eps: float = 1e-8,
) -> Dict[str, float]:
    s0 = scores["s0"]
    s_empty = scores["s_empty"]
    s_del = np.array(scores["s_del"], dtype=np.float64)
    s_rand = np.array(scores.get("s_rand", []), dtype=np.float64)
    delta = s0 - s_del
    aopc = float(delta.mean())
    if s_rand.size == s_del.size and s_rand.size > 0:
        delta_rand = s0 - s_rand
        aopc_rand = float(delta_rand.mean())
    else:
        aopc_rand = None
    delta_max = float(s0 - s_empty)
    delta_scale = float(max(abs(delta_max), eps))
    n_aopc_signed = float(aopc / delta_scale)
    n_aopc = float(np.clip(n_aopc_signed, -1.0, 1.0))
    gap_n = None if aopc_rand is None else float((aopc - aopc_rand) / delta_scale)
    flag = bool(delta_max <= 0)
    flag_small = bool(abs(delta_max) <= eps)
    return {
        "aopc": aopc,
        "aopc_rand": aopc_rand,
        "delta_max": delta_max,
        "delta_scale": delta_scale,
        "n_aopc": n_aopc,
        "n_aopc_signed": n_aopc_signed,
        "gap_aopc": None if aopc_rand is None else float(aopc - aopc_rand),
        "gap_n_aopc": gap_n,
        "flag_delta_max_nonpos": flag,
        "flag_delta_max_small": flag_small,
    }

def compute_comp_suff(
    scores: Dict[str, object],
    m_values: List[int],
    eps: float = 1e-8,
    m_star: int = 32,
) -> Dict[str, float]:
    s0 = scores["s0"]
    s_empty = scores["s_empty"]
    s_del = np.array(scores["s_del"], dtype=np.float64)
    s_keep = np.array(scores["s_keep"], dtype=np.float64)
    delta_max = float(s0 - s_empty)

    comp_vals = s0 - s_del

    comp = float(comp_vals.mean()) if s_del.size > 0 else None
    delta_scale = float(max(abs(delta_max), eps))
    n_comp = float(comp / delta_scale) if comp is not None else None

    if s_keep.size > 0:
        suff_vals = s0 - s_keep
        suff = float(suff_vals.mean())
        n_suff = float(suff / delta_scale)
    else:
        suff_vals = np.array([], dtype=np.float64)
        suff = None
        n_suff = None

    # Comp@m*, Suff@m*
    if m_star in m_values:
        idx = m_values.index(m_star)
        comp_m = float(comp_vals[idx]) if s_del.size > idx else float("nan")
        suff_m = float(suff_vals[idx]) if suff_vals.size > idx else float("nan")
    else:
        comp_m = float("nan")
        suff_m = float("nan")

    return {
        "comp": comp,
        "suff": suff,
        "n_comp": n_comp,
        "n_suff": n_suff,
        "comp_at_m": comp_m,
        "suff_at_m": suff_m,
        "delta_max": delta_max,
        "delta_scale": delta_scale,
        "flag_delta_max_nonpos": bool(delta_max <= 0),
        "flag_delta_max_small": bool(abs(delta_max) <= eps),
    }


def summarize_public_scores(scores):
    values=[scores['s0'],scores['s_empty'],*scores['s_del']]
    if len(scores['s_del'])!=len(M_VALUES) or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Invalid raw causal scores')
    a=compute_aopc_and_normalized(scores,M_VALUES)
    c=compute_comp_suff(dict(scores,s_keep=[]),M_VALUES)
    return dict(n_aopc=a['n_aopc'],n_comp=c['n_comp'],n_aopc_unclipped=a['n_aopc_signed'],
        n_comp_at_32=c['comp_at_m']/a['delta_scale'],delta_max=a['delta_max'],denominator=a['delta_scale'],
        norm_valid=a['delta_max']>.1)


def summarize(s0,empty,deleted):
    return summarize_public_scores(dict(s0=s0,s_empty=empty,s_del=deleted))


def aggregate(rows,profile='medical'):
    if not rows:raise ValueError('No evaluation rows')
    values=[]
    for row in rows:
        if row.get('metric_version')!=version(profile) or row.get('m_values')!=M_VALUES:
            raise ValueError('Evaluation metric profile differs')
        values.append(summarize_public_scores(row))
    valid=[v for v in values if v['norm_valid']]
    return dict(metric_version=version(profile),n_eval=len(rows),n_valid=len(valid),
        n_aopc=statistics.mean(v['n_aopc'] for v in valid) if valid else None,
        n_comp=statistics.mean(v['n_comp'] for v in valid) if valid else None)


@torch.inference_mode()
def score_prompt(model,explainer,record,layer=8,profile='medical'):
    """Rank the top64 features with scalar full-context interventions."""
    device=model.W_U.device
    def full_hook(name):return name if name.startswith('blocks.') else f'blocks.{layer}.'+name
    input_hook=full_hook(explainer.input_hook_point)
    output_hook=full_hook(explainer.output_hook_point)
    tokens=torch.tensor([record['tokens']],device=device,dtype=torch.long)
    logits,cache=model.run_with_cache(tokens,names_filter=list({input_hook,output_hook}))
    target=int(logits[0,-1].argmax())
    _,z=explainer(cache[input_hook][:,-1],return_z=True)
    def scalar(features):
        override=explainer.decoder(features).clone()
        override[features.count_nonzero(-1)==0]=0.
        def hook(activation,hook):
            result=activation.clone();result[:,-1]=override;return result
        with model.hooks(fwd_hooks=[(output_hook,hook)]):value=model(tokens)
        return float(value[0,-1,target])
    s0=scalar(z);empty=scalar(torch.zeros_like(z))
    order=_rank_features_single_example(z,scalar,s0,rank_mode='causal_topz',rank_top_f=64)
    deleted=[]
    for m in M_VALUES:
        features=z.clone();features[:,order[:m]]=0.;deleted.append(scalar(features))
    keys=['record_id','leakage_group','text_sha256','source_id','group_id','token_hash','kind','role','measurement_role']
    row={k:record[k] for k in keys if k in record}
    row.update(target_argmax=target,s0=s0,s_empty=empty,s_del=deleted,m_values=M_VALUES,
        metric_version=version(profile),deletion_feature_ids=order[:128].tolist(),active_features=int(z.count_nonzero()),
        intervention_hook=output_hook,ranking_backend='public_scalar_full_logits',
        actual_deleted_counts=[int(z[0,order[:m]].count_nonzero()) for m in M_VALUES])
    row.update(summarize_public_scores(row))
    return row

"""
Evaluation metrics for Time-shift OOD experiments.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformer_lens import HookedTransformer
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import extract_hook_activations, extract_residual_stream_activations
from utils import (
    get_true_logits_from_model,
    forward_with_override_and_get_target_logit,
    forward_with_override_logits,
    validate_logits_shape,
)
from gae import adapt_features_for_decoder
from ood_utils.saelens_adapters import EXPLAINER_TYPES
from config import Config

FAITH_EMPTY_MODES = ("zero_resid", "decoder_bias")
FAITH_RANK_MODES = ("activation", "causal_topz")
FAITH_HOOK_SITE_MODES = ("auto", "input_hook", "output_hook", "resid_post")


def _resolve_faith_hook_site(explainer, hook_site_mode: str = "auto") -> str:
    if hook_site_mode not in FAITH_HOOK_SITE_MODES:
        raise ValueError(
            f"Unknown hook_site_mode='{hook_site_mode}'. Use one of {FAITH_HOOK_SITE_MODES}."
        )
    input_hook = getattr(explainer, "input_hook_point", None)
    output_hook = getattr(explainer, "output_hook_point", None)

    if hook_site_mode == "auto":
        hook_site = output_hook or input_hook
    elif hook_site_mode == "input_hook":
        hook_site = input_hook
    elif hook_site_mode == "output_hook":
        hook_site = output_hook or input_hook
    else:
        hook_site = "resid_post"

    if hook_site is None:
        hook_site = "resid_post"
    return hook_site


def _extract_faith_features(
    explainer,
    h: torch.Tensor,
    device: torch.device,
    h_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gae_apply_to = getattr(explainer, "gae_apply_to", None)
    with torch.no_grad():
        if gae_apply_to == "encoder" and hasattr(explainer, "B_gae"):
            return torch.relu(h @ explainer.B_gae.to(device))
        _, z = explainer(h, return_z=True)
        if gae_apply_to == "decoder" and hasattr(explainer, "B_gae"):
            adapter_mode = getattr(explainer, "gae_feature_adapter", None)
            if adapter_mode:
                h_fit = h_target if h_target is not None else h
                return adapt_features_for_decoder(
                    z0=z,
                    h=h_fit,
                    decoder=explainer.B_gae.to(device),
                    adapter_mode=adapter_mode,
                    gains=getattr(explainer, "gae_feature_gains", None),
                    ridge=float(getattr(explainer, "gae_support_refit_ridge", 1e-3)),
                    max_support=int(getattr(explainer, "gae_support_refit_topk", 128)),
                )
        return z


def _resolve_decoder_override(explainer, device: torch.device):
    decoder_override = getattr(explainer, "gae_decoder_override", None)
    decoder_bias = getattr(explainer, "gae_decoder_bias", None)
    if decoder_override is not None:
        decoder_override = decoder_override.to(device)
        if decoder_bias is not None:
            decoder_bias = decoder_bias.to(device)
        return decoder_override, decoder_bias

    gae_apply_to = getattr(explainer, "gae_apply_to", None)
    if gae_apply_to == "decoder" and hasattr(explainer, "B_gae"):
        decoder_override = explainer.B_gae.to(device)
        decoder_bias = getattr(explainer, "gae_decoder_bias", None)
        if decoder_bias is not None:
            decoder_bias = decoder_bias.to(device)
        return decoder_override, decoder_bias
    if hasattr(explainer, "D_gae"):
        return explainer.D_gae.to(device), None
    return None, None


def _decode_features(
    explainer,
    feats: torch.Tensor,
    h_template: torch.Tensor,
    decoder_override: Optional[torch.Tensor] = None,
    decoder_bias: Optional[torch.Tensor] = None,
    empty_mode: str = "zero_resid",
) -> torch.Tensor:
    if decoder_override is not None:
        h_recon = feats @ decoder_override.T
        if decoder_bias is not None:
            h_recon = h_recon + decoder_bias
    else:
        if getattr(explainer, "output_kind", None) == "logits":
            raise RuntimeError(
                "Explainer output_kind=logits cannot be used for TRUE-logit evaluation. "
                "Provide a decoder that reconstructs hidden activations (d_model)."
            )
        if not hasattr(explainer, "decoder"):
            raise RuntimeError("Explainer does not provide a decoder to reconstruct hidden activations.")
        h_recon = explainer.decoder(feats)

    if empty_mode == "zero_resid":
        empty_rows = torch.count_nonzero(feats, dim=1) == 0
        if torch.any(empty_rows):
            h_recon = h_recon.clone()
            h_recon[empty_rows] = 0.0
    return h_recon


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


def compute_ablation_auc(
    model: HookedTransformer,
    explainer,
    tokens: torch.Tensor,
    activations: torch.Tensor,
    logits_original: torch.Tensor,
    target_token_idx: int,
    layer: int,
    activations_out: Optional[torch.Tensor] = None,
    model_type: str = 'transcoder',
    m_values: List[int] = None,
    device=None
) -> float:
    """
    Compute ablation AUC for causal faithfulness.
    
    For each input example, compute ablation effect at different feature retention levels.
    AUC is computed over normalized ablation effects.
    
    Args:
        model: HookedTransformer instance
        explainer: Transcoder or SAE instance (or adapted dictionary B)
        tokens: Input tokens, shape [batch_size, seq_len]
        activations: Residual stream activations, shape [batch_size, d_model]
        logits_original: Original logits, shape [batch_size, vocab_size]
        target_token_idx: Target token index (self-consistency target)
        layer: Layer index
        model_type: 'transcoder' or 'sae'
        m_values: List of top-m features to retain (default: [0, 10, 20, ..., k])
        device: Device to use
    
    Returns:
        auc: Mean ablation AUC across examples
    """
    if device is None:
        device = activations.device

    model.eval()
    if isinstance(explainer, EXPLAINER_TYPES):
        explainer.eval()

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    tokens = tokens.to(device)
    activations = activations.to(device)
    logits_original = logits_original.to(device)

    batch_size = activations.shape[0]

    if isinstance(explainer, EXPLAINER_TYPES):
        z = _extract_faith_features(explainer, activations, device, h_target=activations_out)
    else:
        raise NotImplementedError("Direct dictionary ablation not yet implemented")

    if m_values is None:
        m_values = list(range(0, min(z.shape[1], 200), 10))

    if isinstance(target_token_idx, torch.Tensor):
        target_ids = target_token_idx.to(device=device, dtype=torch.long)
    elif isinstance(target_token_idx, (list, tuple, np.ndarray)):
        target_ids = torch.tensor(target_token_idx, device=device, dtype=torch.long)
    else:
        target_ids = torch.full(
            (batch_size,),
            int(target_token_idx),
            dtype=torch.long,
            device=device,
        )

    if target_ids.shape[0] != batch_size:
        raise ValueError(
            f"target_token_idx batch mismatch: got {target_ids.shape[0]}, expected {batch_size}"
        )

    D_override, decoder_bias = _resolve_decoder_override(explainer, device)
    aucs = _compute_batch_auc_from_features(
        model=model,
        explainer=explainer,
        tokens_batch=tokens,
        layer=layer,
        position=-1,
        z=z,
        logits_original=logits_original,
        target_token_idx=target_ids,
        model_type=model_type,
        m_values=m_values,
        D=D_override,
        decoder_bias=decoder_bias,
    )

    return float(np.mean(aucs)) if len(aucs) > 0 else float("nan")


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Compute Spearman rank correlation between 1D arrays x and y.
    Uses argsort-based ranking (no special tie handling).
    """
    if x.size != y.size or x.size == 0:
        return None
    # Ranks via double argsort
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    # Pearson correlation of ranks
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = np.linalg.norm(vx) * np.linalg.norm(vy)
    if denom == 0:
        return None
    return float(np.dot(vx, vy) / denom)


def _compute_batch_auc_from_features(
    model: HookedTransformer,
    explainer,
    tokens_batch: torch.Tensor,  # [B, L]
    layer: int,
    position: int,
    z: torch.Tensor,             # [B, K]
    logits_original: torch.Tensor,  # [B, V]
    target_token_idx: torch.Tensor,  # [B]
    model_type: str,
    m_values: Optional[List[int]] = None,
    D: Optional[torch.Tensor] = None,
    decoder_bias: Optional[torch.Tensor] = None,
    empty_mode: str = "zero_resid",
    rank_mode: str = "activation",
    rank_top_f: int = 64,
) -> List[float]:
    """
    Batch AUC computation using activation override and TRUE logits.
    Returns list of per-example AUCs.
    """
    B, K = z.shape
    device = z.device
    if empty_mode not in FAITH_EMPTY_MODES:
        raise ValueError(f"Unknown empty_mode='{empty_mode}'. Use one of {FAITH_EMPTY_MODES}.")

    if m_values is None:
        m_grid = list(range(0, min(K, 200), 10))
    else:
        m_grid = list(m_values)

    if rank_mode not in FAITH_RANK_MODES:
        raise ValueError(f"Unknown rank_mode='{rank_mode}'. Use one of {FAITH_RANK_MODES}.")

    # Resolve hook site
    hook_site = getattr(explainer, "output_hook_point", None) or getattr(explainer, "input_hook_point", None)
    if hook_site is None:
        hook_site = "resid_post"

    # Precompute "all-features reconstruction" baseline logits (not raw model logits).
    h_template = torch.zeros((B, model.cfg.d_model), device=device, dtype=z.dtype)
    h_all = _decode_features(
        explainer=explainer,
        feats=z,
        h_template=h_template,
        decoder_override=D,
        decoder_bias=decoder_bias,
        empty_mode=empty_mode,
    )
    logits_all = forward_with_override_logits(
        model=model,
        input_ids=tokens_batch,
        layer=layer,
        position=position,
        hook_site=hook_site,
        h_override=h_all,
    )
    validate_logits_shape(logits_all, expected_vocab_size=model.cfg.d_vocab)
    base_logit = logits_all[:, position, :][torch.arange(B, device=device), target_token_idx]  # [B]

    # Feature ranking
    if rank_mode == "activation":
        top_idx = torch.argsort(z, dim=1, descending=True)  # [B, K]
    else:
        # causal_topz: per-example ranking with top-F causal effects (uses all-features baseline)
        top_idx_rows = []
        for b in range(B):
            z_b = z[b:b+1]
            tgt_b = int(target_token_idx[b].item())
            token_b = tokens_batch[b:b+1]

            def _score_fn(feats_b: torch.Tensor) -> float:
                h_recon_b = _decode_features(
                    explainer=explainer,
                    feats=feats_b,
                    h_template=h_template[:1],
                    decoder_override=D,
                    decoder_bias=decoder_bias,
                    empty_mode=empty_mode,
                )
                target_logit_b, _ = forward_with_override_and_get_target_logit(
                    model=model,
                    input_ids=token_b,
                    layer=layer,
                    position=position,
                    hook_site=hook_site,
                    h_override=h_recon_b,
                    target_id=tgt_b,
                    return_logits=False,
                )
                return float(target_logit_b.item())

            ranked = _rank_features_single_example(
                z=z_b,
                target_logit_from_features_fn=_score_fn,
                base_score=float(base_logit[b].item()),
                rank_mode=rank_mode,
                rank_top_f=rank_top_f,
            )
            top_idx_rows.append(ranked)
        top_idx = torch.stack(top_idx_rows, dim=0)

    deltas = []
    for m in m_grid:
        m_eff = int(max(0, min(int(m), K)))
        if m_eff == 0:
            feats = torch.zeros_like(z)
        elif m_eff >= K:
            feats = z.clone()
        else:
            feats = torch.zeros_like(z)
            sel = top_idx[:, :m_eff]
            feats.scatter_(1, sel, z.gather(1, sel))

        h_recon = _decode_features(
            explainer=explainer,
            feats=feats,
            h_template=h_template,
            decoder_override=D,
            decoder_bias=decoder_bias,
            empty_mode=empty_mode,
        )

        logits_abl = forward_with_override_logits(
            model=model,
            input_ids=tokens_batch,
            layer=layer,
            position=position,
            hook_site=hook_site,
            h_override=h_recon,
        )
        validate_logits_shape(logits_abl, expected_vocab_size=model.cfg.d_vocab)
        target_logit_abl = logits_abl[:, position, :][torch.arange(B, device=device), target_token_idx]
        deltas.append((base_logit - target_logit_abl).detach().cpu().numpy())

    deltas_np = np.stack(deltas, axis=1)  # [B, M]
    sigma_delta = 1.0 / (1.0 + np.exp(-deltas_np))
    aucs = sigma_delta.mean(axis=1)
    return [float(x) for x in aucs]


def _compute_single_example_metrics(
    explainer,
    model: HookedTransformer,
    tokens: torch.Tensor,            # [1, seq_len]
    layer: int,
    position: int,
    h: torch.Tensor,                 # [1, d_model] (input)
    logits_original: torch.Tensor,   # [1, V]
    target_token_idx: int,
    model_type: str,
    device: torch.device,
    h_out: Optional[torch.Tensor] = None,  # [1, d_model] (output)
    m_values: Optional[List[int]] = None,
    k_list: Optional[List[int]] = None,
    tau_abs: float = 1.0,
    tau_rel_frac: float = 0.3,
    max_rank_features: int = 200,
    precomputed_auc: Optional[float] = None,
    empty_mode: str = "zero_resid",
    rank_mode: str = "causal_topz",
    rank_top_f: int = 64,
    hook_site_mode: str = "auto",
) -> Dict[str, object]:
    """
    Compute per-example Time-shift OOD metrics:
    - Ablation AUC
    - Δ@K (for each K in k_list)
    - m@τ (absolute / relative)
    - Spearman ρ between explainer scores and per-feature causal effects.
    """
    h = h.to(device)
    h_ref = h_out.to(device) if h_out is not None else h
    logits_original = logits_original.to(device)
    tokens = tokens.to(device)
    if empty_mode not in FAITH_EMPTY_MODES:
        raise ValueError(f"Unknown empty_mode='{empty_mode}'. Use one of {FAITH_EMPTY_MODES}.")
    if rank_mode not in FAITH_RANK_MODES:
        raise ValueError(f"Unknown rank_mode='{rank_mode}'. Use one of {FAITH_RANK_MODES}.")
    if hook_site_mode not in FAITH_HOOK_SITE_MODES:
        raise ValueError(
            f"Unknown hook_site_mode='{hook_site_mode}'. Use one of {FAITH_HOOK_SITE_MODES}."
        )

    # Determine hook site for override
    hook_site = _resolve_faith_hook_site(explainer, hook_site_mode=hook_site_mode)

    # Feature extraction z: [1,k].
    # Theory decoder mode preserves the original encoder; legacy encoder mode still rotates features.
    if isinstance(explainer, EXPLAINER_TYPES):
        z = _extract_faith_features(explainer, h, device, h_target=h_ref)
    else:
        raise NotImplementedError("Direct dictionary ablation not implemented for raw dictionaries.")

    k = z.shape[1]

    # Decoder override selection (D, bias)
    decoder_override, decoder_bias = _resolve_decoder_override(explainer, device)

    def _metrics_for_decoder(D: Optional[torch.Tensor]) -> Dict[str, object]:
        """
        Compute all metrics for a given decoder matrix D.
        If D is None, use original explainer decoder (W_tilde / SAE decoder).
        """
        def target_logit_from_hidden(h_override: torch.Tensor) -> float:
            target_logit, _ = forward_with_override_and_get_target_logit(
                model=model,
                input_ids=tokens,
                layer=layer,
                position=position,
                hook_site=hook_site,
                h_override=h_override,
                target_id=target_token_idx,
                return_logits=False,
            )
            return float(target_logit.item())

        def target_logit_from_features(features: torch.Tensor) -> float:
            """
            Given feature vector [1,k], reconstruct hidden and return target logit
            from TRUE model logits via activation override.
            """
            if empty_mode == "zero_resid" and torch.count_nonzero(features).item() == 0:
                h_recon = torch.zeros_like(h_ref)
            else:
                h_recon = _decode_features(
                    explainer=explainer,
                    feats=features,
                    h_template=h_ref,
                    decoder_override=D,
                    decoder_bias=decoder_bias,
                    empty_mode=empty_mode,
                )
            return target_logit_from_hidden(h_recon)

        # Compute baselines:
        # - target_logit_true: z(h)_t from original model (for diagnostics only)
        # - target_logit_all_recon: z(hat(h)(all))_t (faithfulness baseline)
        target_logit_true = logits_original[0, target_token_idx].item()
        z_all = z.clone()  # All features
        target_logit_all_recon = target_logit_from_features(z_all.to(device))
        recon_gap = float(target_logit_true - target_logit_all_recon)

        # Default grids
        if m_values is None:
            m_grid = list(range(0, min(k, 200), 10))
        else:
            m_grid = list(m_values)
        if k_list is None:
            # Only K=50 is needed for Suff/Comp summary.
            k_grid = [50]
        else:
            k_grid = list(k_list)

        top_idx = _rank_features_single_example(
            z=z,
            target_logit_from_features_fn=target_logit_from_features,
            base_score=target_logit_all_recon,
            rank_mode=rank_mode,
            rank_top_f=rank_top_f,
        )
        feature_scores = z[0]

        # --- Hidden-space AUC ---
        hidden_auc = None
        if getattr(explainer, "output_kind", None) != "logits" and (D is not None or hasattr(explainer, "decoder")):
            with torch.no_grad():
                h_all = _decode_features(
                    explainer=explainer,
                    feats=z.to(device),
                    h_template=h_ref,
                    decoder_override=D,
                    decoder_bias=decoder_bias,
                    empty_mode=empty_mode,
                )
                err_full = torch.norm(h_ref - h_all, dim=-1).item()
                deltas_h = []
                for m in m_grid:
                    m_eff = int(max(0, min(int(m), k)))
                    if m_eff == 0:
                        feats = torch.zeros_like(z)
                    elif m_eff >= k:
                        feats = z.clone()
                    else:
                        feats = torch.zeros_like(z)
                        sel = top_idx[:m_eff]
                        feats[0, sel] = z[0, sel]
                    h_rec = _decode_features(
                        explainer=explainer,
                        feats=feats.to(device),
                        h_template=h_ref,
                        decoder_override=D,
                        decoder_bias=decoder_bias,
                        empty_mode=empty_mode,
                    )
                    err_m = torch.norm(h_ref - h_rec, dim=-1).item()
                    deltas_h.append(err_m - err_full)
                hidden_auc = float(torch.sigmoid(torch.tensor(deltas_h)).mean().item())

        # --- 1) Ablation AUC (keep-top-m) ---
        # Uses z(hat(h)(all))_t as baseline to avoid reconstruction-error leakage.
        if precomputed_auc is not None:
            auc_keep_val = float(precomputed_auc)
        else:
            ablation_effects = []
            with torch.no_grad():
                for m in m_grid:
                    m_eff = int(max(0, min(int(m), k)))
                    if m_eff == 0:
                        feats = torch.zeros_like(z)
                    elif m_eff >= k:
                        feats = z.clone()
                    else:
                        feats = torch.zeros_like(z)
                        sel = top_idx[:m_eff]
                        feats[0, sel] = z[0, sel]

                    target_logit_abl = target_logit_from_features(feats.to(device))
                    delta_m = target_logit_all_recon - target_logit_abl
                    ablation_effects.append(delta_m)

            ablation_effects_np = np.array(ablation_effects, dtype=np.float64)
            sigma_delta = torch.sigmoid(torch.from_numpy(ablation_effects_np)).cpu().numpy()
            auc_keep_val = float(sigma_delta.mean())

        # delete-top-m AUC is skipped for speed (not part of the retained metric set).
        auc_del_val = None

        # --- 2) Δ@K (internal, used for Comp/CF) ---
        # Comp(K) = z(hat(h)(all))_t - z(hat(h)(all \ S_K))_t (Section 6.6)
        delta_at_k_local = {}
        with torch.no_grad():
            for K in k_grid:
                K_eff = int(max(0, min(int(K), k)))
                if K_eff == 0:
                    delta_k = 0.0
                else:
                    feats = z.clone()
                    abl = top_idx[:K_eff]
                    feats[0, abl] = 0.0
                    target_logit_abl = target_logit_from_features(feats.to(device))
                    # Use z(hat(h)(all))_t as baseline for Comp(K) per Timeshift-OOD.md
                    delta_k = target_logit_all_recon - target_logit_abl
                delta_at_k_local[int(K)] = float(delta_k)

        # --- 3) m@τ --- (disabled for speed)
        # m_at_tau_abs_local = None
        # tau_rel = tau_rel_frac * max(target_logit_all_recon, 0.0)
        # m_at_tau_rel_local = None
        #
        # with torch.no_grad():
        #     max_m_search = min(k, max(m_grid) if len(m_grid) > 0 else k)
        #     for m in range(1, max_m_search + 1):
        #         feats = z.clone()
        #         abl = top_idx[:m]
        #         feats[0, abl] = 0.0
        #         target_logit_abl = target_logit_from_features(feats.to(device))
        #         # Use z(hat(h)(all))_t as baseline per Timeshift-OOD.md Section 6.3
        #         delta_m = target_logit_all_recon - target_logit_abl
        #
        #         if m_at_tau_abs_local is None and delta_m >= tau_abs:
        #             m_at_tau_abs_local = m
        #         if m_at_tau_rel_local is None and delta_m >= tau_rel:
        #             m_at_tau_rel_local = m
        #         if m_at_tau_abs_local is not None and m_at_tau_rel_local is not None:
        #             break
        #
        # if m_at_tau_abs_local is None:
        #     m_at_tau_abs_local = max_m_search + 1
        # if m_at_tau_rel_local is None:
        #     m_at_tau_rel_local = max_m_search + 1
        m_at_tau_abs_local = None
        m_at_tau_rel_local = None

        # --- 4) Spearman ρ --- (disabled for speed)
        spearman_rho_local = None
        # with torch.no_grad():
        #     F = min(max_rank_features, k)
        #     if F > 1:
        #         top_feat_idx = top_idx[:F]
        #         scores_sub = feature_scores[top_feat_idx].detach().cpu().numpy()
        #         effects_sub = []
        #         for j in top_feat_idx:
        #             feats = z.clone()
        #             feats[0, j] = 0.0
        #             target_logit_abl = target_logit_from_features(feats.to(device))
        #             # Use z(hat(h)(all))_t as baseline per Timeshift-OOD.md Section 6.4
        #             e_j = target_logit_all_recon - target_logit_abl
        #             effects_sub.append(e_j)
        #         effects_sub = np.array(effects_sub, dtype=np.float64)
        #         spearman_rho_local = _spearman_corr(scores_sub, effects_sub)

        return {
            # Backward-compatible key: keep-top-m AUC.
            "auc": auc_keep_val,
            "auc_keep": auc_keep_val,
            "auc_del": auc_del_val,
            "auc_hidden": hidden_auc,
            # Δ@K is kept internal for comp/CF computation but not aggregated as a separate metric.
            "delta_at_k": delta_at_k_local,
            "top_idx": top_idx,
            "m_at_tau_abs": int(m_at_tau_abs_local) if m_at_tau_abs_local is not None else None,
            "m_at_tau_rel": int(m_at_tau_rel_local) if m_at_tau_rel_local is not None else None,
            "spearman_rho": spearman_rho_local,
            "target_logit_true": float(target_logit_true),
            "target_logit_recon_all": float(target_logit_all_recon),
            "recon_gap": float(recon_gap),
        }

    base_metrics = _metrics_for_decoder(D=decoder_override)

    result = {
        "auc": base_metrics["auc"],
        "auc_keep": base_metrics.get("auc_keep"),
        "auc_del": base_metrics.get("auc_del"),
        "auc_hidden": base_metrics["auc_hidden"],
        "delta_at_k": base_metrics["delta_at_k"],
        "m_at_tau_abs": base_metrics["m_at_tau_abs"],
        "m_at_tau_rel": base_metrics["m_at_tau_rel"],
        "spearman_rho": base_metrics["spearman_rho"],
        "target_logit_true": base_metrics.get("target_logit_true"),
        "target_logit_recon_all": base_metrics.get("target_logit_recon_all"),
        "recon_gap": base_metrics.get("recon_gap"),
        # Reused by compute_feature_mask_scores to avoid duplicate ranking/feature extraction.
        "faith_z": z.detach(),
        "faith_top_idx": (
            base_metrics.get("top_idx").detach()
            if base_metrics.get("top_idx") is not None
            else None
        ),
        "faith_s0": base_metrics.get("target_logit_recon_all"),
        # Placeholders for sufficiency / comprehensiveness / CF / FT (filled below)
        "suff": None,
        "comp": None,
        "cf": None,
        "ft_ratio": None,
    }

    # --- 5) Sufficiency / Comprehensiveness ---
    # We compute these at a fixed budget K (e.g., 50 or less if k is smaller).
    K_faith = min(50, k)
    if K_faith > 0:
        # Helper to compute suff/comp from per-decoder metrics.
        def _suff_comp(metrics_dict, D: Optional[torch.Tensor]):
            delta_at_k_dict = metrics_dict["delta_at_k"]
            delta_k = delta_at_k_dict.get(K_faith, 0.0)
            top_idx_local = metrics_dict.get("top_idx")
            if top_idx_local is None:
                top_idx_local = torch.argsort(z[0], descending=True)

            def target_logit_from_features_local(features: torch.Tensor) -> float:
                if empty_mode == "zero_resid" and torch.count_nonzero(features).item() == 0:
                    h_recon = torch.zeros_like(h_ref)
                else:
                    h_recon = _decode_features(
                        explainer=explainer,
                        feats=features,
                        h_template=h_ref,
                        decoder_override=D,
                        decoder_bias=decoder_bias,
                        empty_mode=empty_mode,
                    )

                target_logit, _ = forward_with_override_and_get_target_logit(
                    model=model,
                    input_ids=tokens,
                    layer=layer,
                    position=position,
                    hook_site=hook_site,
                    h_override=h_recon,
                    target_id=target_token_idx,
                    return_logits=False,
                )
                return float(target_logit.item())

            # Build features vectors for S_K (keep only top-K) and all\setminus S_K
            feats_keep = torch.zeros_like(z)
            feats_keep[0, top_idx_local[:K_faith]] = z[0, top_idx_local[:K_faith]]

            # Sufficiency: Suff(K) = z(hat(h)(S_K))_t (Section 6.6)
            suff_val = float(target_logit_from_features_local(feats_keep.to(device)))
            # Comprehensiveness: Comp(K) = z(hat(h)(all))_t - z(hat(h)(all \ S_K))_t (Section 6.6)
            comp_val = float(delta_k)  # delta_k already uses target_logit_all_recon as baseline
            return suff_val, comp_val

        # Base decoder
        base_suff, base_comp = _suff_comp(base_metrics, decoder_override)
        result["suff"] = base_suff
        result["comp"] = base_comp
        result["cf"] = None

    # Faithfulness transfer (ID vs OOD) requires ID stats; left as None here.

    return result


def compute_feature_mask_scores(
    explainer,
    model: HookedTransformer,
    tokens: torch.Tensor,            # [1, seq_len]
    layer: int,
    position: int,
    h: torch.Tensor,                 # [1, d_model]
    logits_original: torch.Tensor,   # [1, V]
    target_token_idx: int,
    model_type: str,
    m_values: List[int],
    h_out: Optional[torch.Tensor] = None,  # [1, d_model]
    random_repeats: int = 5,
    max_rank_features: int = 200,
    empty_mode: str = "zero_resid",
    rank_mode: str = "causal_topz",
    rank_top_f: int = 64,
    hook_site_mode: str = "auto",
    compute_random_baseline: bool = True,
    compute_sufficiency: bool = True,
    precomputed_z: Optional[torch.Tensor] = None,
    precomputed_top_idx: Optional[torch.Tensor] = None,
    precomputed_s0: Optional[float] = None,
    precomputed_s_empty: Optional[float] = None,
) -> Dict[str, object]:
    """
    Compute per-example scores for deletion (remove top-m) and sufficiency (keep top-m),
    plus random deletion baseline. Uses the SAME reconstruction logic as existing metrics.
    """
    if empty_mode not in FAITH_EMPTY_MODES:
        raise ValueError(f"Unknown empty_mode='{empty_mode}'. Use one of {FAITH_EMPTY_MODES}.")
    if rank_mode not in FAITH_RANK_MODES:
        raise ValueError(f"Unknown rank_mode='{rank_mode}'. Use one of {FAITH_RANK_MODES}.")
    if hook_site_mode not in FAITH_HOOK_SITE_MODES:
        raise ValueError(
            f"Unknown hook_site_mode='{hook_site_mode}'. Use one of {FAITH_HOOK_SITE_MODES}."
        )

    device = logits_original.device
    h = h.to(device)
    tokens = tokens.to(device)
    if precomputed_z is not None:
        z = precomputed_z.to(device)
    else:
        if isinstance(explainer, EXPLAINER_TYPES):
            z = _extract_faith_features(
                explainer,
                h,
                device,
                h_target=h_out if h_out is not None else h,
            )
        else:
            raise NotImplementedError("Direct dictionary ablation not implemented for raw dictionaries.")

    k = z.shape[1]

    # Decoder override for feature masking scores
    _gae_decoder, _gae_decoder_bias = _resolve_decoder_override(explainer, device)

    # Helper: compute target logit from features via TRUE logits
    hook_site = _resolve_faith_hook_site(explainer, hook_site_mode=hook_site_mode)

    def _target_logit_from_hidden(h_override: torch.Tensor) -> float:
        target_logit, _ = forward_with_override_and_get_target_logit(
            model=model,
            input_ids=tokens,
            layer=layer,
            position=position,
            hook_site=hook_site,
            h_override=h_override,
            target_id=target_token_idx,
            return_logits=False,
        )
        return float(target_logit.item())

    def _target_logit_from_features(feats: torch.Tensor) -> float:
        if empty_mode == "zero_resid" and torch.count_nonzero(feats).item() == 0:
            h_recon = torch.zeros_like(h)
        else:
            h_recon = _decode_features(
                explainer=explainer,
                feats=feats,
                h_template=h,
                decoder_override=_gae_decoder,
                decoder_bias=_gae_decoder_bias,
                empty_mode=empty_mode,
            )
        return _target_logit_from_hidden(h_recon)

    s_true = float(logits_original[0, target_token_idx].item())
    feats_keep_all = z.clone()
    if precomputed_s0 is not None:
        s_recon = float(precomputed_s0)
    else:
        s_recon = float(_target_logit_from_features(feats_keep_all))
    # Use reconstruction baseline to isolate causal deletion effects from reconstruction error.
    s0 = s_recon
    recon_gap = float(s_true - s_recon)

    if precomputed_top_idx is not None:
        top_idx = precomputed_top_idx.to(device)
    else:
        top_idx = _rank_features_single_example(
            z=z,
            target_logit_from_features_fn=_target_logit_from_features,
            base_score=s_recon,
            rank_mode=rank_mode,
            rank_top_f=rank_top_f,
        )
    # s_empty
    if precomputed_s_empty is not None:
        s_empty = float(precomputed_s_empty)
    else:
        empty_feats = torch.zeros_like(z)
        s_empty = _target_logit_from_features(empty_feats)

    s_del = []
    s_keep = []
    s_rand = []

    for m in m_values:
        if m <= 0:
            # delete none => keep all
            feats_keep_all = z.clone()
            s_del.append(_target_logit_from_features(feats_keep_all))
            # keep none
            if compute_sufficiency:
                s_keep.append(s_empty)
            if compute_random_baseline:
                s_rand.append(s_del[-1])
            continue
        if m >= k:
            # delete all
            s_del.append(s_empty)
            # keep all
            if compute_sufficiency:
                s_keep.append(_target_logit_from_features(z))
            if compute_random_baseline:
                s_rand.append(s_empty)
            continue

        # Deletion: remove top-m, keep complement
        top_m = top_idx[:m]
        feats_del = z.clone()
        feats_del[0, top_m] = 0.0
        s_del.append(_target_logit_from_features(feats_del))

        # Sufficiency: keep top-m only
        if compute_sufficiency:
            feats_keep = torch.zeros_like(z)
            feats_keep[0, top_m] = z[0, top_m]
            s_keep.append(_target_logit_from_features(feats_keep))

        # Random deletion baseline
        if compute_random_baseline:
            rand_vals = []
            for _ in range(random_repeats):
                perm = torch.randperm(k, device=z.device)
                rand_sel = perm[:m]
                feats_rand = z.clone()
                feats_rand[0, rand_sel] = 0.0
                rand_vals.append(_target_logit_from_features(feats_rand))
            s_rand.append(float(np.mean(rand_vals)))

    s_del_arr = np.array(s_del, dtype=np.float64) if len(s_del) > 0 else np.array([], dtype=np.float64)
    aopc_like_delta = s0 - s_del_arr if s_del_arr.size > 0 else np.array([], dtype=np.float64)
    aopc_negative_frac = float(np.mean(aopc_like_delta < 0.0)) if aopc_like_delta.size > 0 else float("nan")
    topm_logit_increase_frac = float(np.mean(s_del_arr > s0)) if s_del_arr.size > 0 else float("nan")

    return {
        "s0": s0,
        "s_true": s_true,
        "s_recon": s_recon,
        "recon_gap": recon_gap,
        "s_empty": s_empty,
        "s_del": s_del,
        "s_keep": s_keep,
        "s_rand": s_rand,
        "k": k,
        "empty_mode": empty_mode,
        "rank_mode": rank_mode,
        "rank_top_f": int(rank_top_f),
        "aopc_negative_frac": aopc_negative_frac,
        "topm_logit_increase_frac": topm_logit_increase_frac,
    }


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


def evaluate_causal_faithfulness(
    model: HookedTransformer,
    explainer,
    tokens_list: List[torch.Tensor],
    layer: int,
    model_type: str = 'transcoder',
    device=None,
    target_mode: str = 'argmax',
    batch_id: Optional[List[torch.Tensor]] = None,
    target_token_ids: Optional[List[Optional[int]]] = None,
    faith_m_values: Optional[List[int]] = None,
    faith_random_repeats: int = 5,
    faith_m_star: int = 32,
    faith_empty_mode: str = "zero_resid",
    faith_rank_mode: str = "causal_topz",
    faith_rank_top_f: int = 64,
    faith_hook_site_mode: str = "auto",
    faith_compute_random_baseline: bool = False,
) -> Dict[str, float]:
    """
    Evaluate causal faithfulness via ablation AUC.
    
    Args:
        model: HookedTransformer instance
        explainer: Transcoder or SAE instance (or adapted dictionary)
        tokens_list: List of tokenized sequences (OOD)
        layer: Layer index for activation extraction
        model_type: 'transcoder' or 'sae'
        device: Device to use
        target_mode: 'argmax' (self-consistency), 'gold' (next-token label), or 'provided' (use target_token_ids)
        batch_id: Optional list of ID tokenized sequences for FT computation
        target_token_ids: Optional list of per-example target token IDs (for target_mode='provided')
    
    Returns:
        Dictionary with metric values
    """
    if device is None:
        device = Config.device
    if faith_empty_mode not in FAITH_EMPTY_MODES:
        raise ValueError(
            f"Unknown faith_empty_mode='{faith_empty_mode}'. Use one of {FAITH_EMPTY_MODES}."
        )
    if faith_rank_mode not in FAITH_RANK_MODES:
        raise ValueError(
            f"Unknown faith_rank_mode='{faith_rank_mode}'. Use one of {FAITH_RANK_MODES}."
        )
    if faith_hook_site_mode not in FAITH_HOOK_SITE_MODES:
        raise ValueError(
            f"Unknown faith_hook_site_mode='{faith_hook_site_mode}'. Use one of {FAITH_HOOK_SITE_MODES}."
        )
    if faith_rank_top_f <= 0:
        raise ValueError("faith_rank_top_f must be positive.")
    
    model.to(device)
    if isinstance(explainer, EXPLAINER_TYPES):
        explainer.to(device)
    
    all_aucs = []
    all_m_at_tau_abs = []
    all_m_at_tau_rel = []
    all_spearman = []

    all_suff = []
    all_comp = []
    all_cf = []

    # Optional AOPC / Comp / Suff over M
    all_aopc = []
    all_aopc_rand = []
    all_gap_aopc = []
    all_n_aopc = []
    all_gap_n_aopc = []
    all_comp_m = []
    all_suff_m = []
    all_n_comp_m = []
    all_n_suff_m = []
    all_comp_at_m = []
    all_suff_at_m = []
    all_delta_max = []
    all_recon_gap = []
    all_recon_s_true = []
    all_recon_s_recon = []
    all_aopc_negative_frac = []
    all_topm_logit_increase_frac = []
    all_aucs_del = []
    all_n_aopc_delta_pos = []
    all_n_aopc_delta_nonpos = []
    all_comp_m_delta_pos = []
    all_comp_m_delta_nonpos = []
    all_n_aopc_norm_valid = []
    all_n_comp_norm_valid = []
    all_n_suff_norm_valid = []
    all_n_aopc_norm_invalid = []
    all_n_comp_norm_invalid = []
    all_n_suff_norm_invalid = []
    delta_max_nonpos = 0
    norm_valid_count = 0
    norm_invalid_count = 0
    norm_min_delta = float(getattr(Config, "FAITH_NORM_MIN_DELTA", 0.1))

    all_hidden_aucs = []
    
    # For FT: collect Comp(K) values from OOD
    all_comp_ood = []

    input_hook = getattr(explainer, "input_hook_point", None)
    output_hook = getattr(explainer, "output_hook_point", None)
    eval_position = -2 if target_mode == 'gold' else -1

    # Process each sequence without padding contamination.
    for idx, tokens in tqdm(
        enumerate(tokens_list),
        desc="Evaluating OOD",
        total=len(tokens_list),
    ):
        tokens_batch = tokens.unsqueeze(0).to(device) if tokens.dim() == 1 else tokens.to(device)

        if input_hook:
            activations = extract_hook_activations(
                model, tokens_batch, layer, position=eval_position, hook_name=input_hook
            )
        else:
            activations = extract_residual_stream_activations(
                model, tokens_batch, layer, position=eval_position
            )

        activations_out = None
        if getattr(explainer, "output_kind", None) == "mlp_out" and output_hook:
            activations_out = extract_hook_activations(
                model, tokens_batch, layer, position=eval_position, hook_name=output_hook
            )

        logits_full = get_true_logits_from_model(model, tokens_batch)
        validate_logits_shape(logits_full, expected_vocab_size=model.cfg.d_vocab)
        logits_original = logits_full[:, eval_position, :]

        if target_mode == 'gold':
            tgt_idx = int(tokens_batch[0, -1].item())
        elif target_mode == 'provided':
            if target_token_ids is None:
                raise ValueError("target_mode='provided' requires target_token_ids")
            tid = target_token_ids[idx] if idx < len(target_token_ids) else None
            tgt_idx = int(tid) if tid is not None else int(logits_original[0].argmax(dim=-1).item())
        else:
            tgt_idx = int(logits_original[0].argmax(dim=-1).item())

        metrics = _compute_single_example_metrics(
            explainer=explainer,
            model=model,
            tokens=tokens_batch,
            layer=layer,
            position=eval_position,
            h=activations,
            logits_original=logits_original,
            target_token_idx=tgt_idx,
            model_type=model_type,
            device=device,
            h_out=activations_out,
            precomputed_auc=None,
            empty_mode=faith_empty_mode,
            rank_mode=faith_rank_mode,
            rank_top_f=faith_rank_top_f,
            hook_site_mode=faith_hook_site_mode,
        )
        all_aucs.append(metrics["auc"])
        if metrics.get("auc_del") is not None:
            all_aucs_del.append(metrics["auc_del"])
        if metrics.get("auc_hidden") is not None:
            all_hidden_aucs.append(metrics["auc_hidden"])
        if metrics.get("recon_gap") is not None:
            all_recon_gap.append(float(metrics["recon_gap"]))
        if metrics.get("target_logit_true") is not None:
            all_recon_s_true.append(float(metrics["target_logit_true"]))
        if metrics.get("target_logit_recon_all") is not None:
            all_recon_s_recon.append(float(metrics["target_logit_recon_all"]))

        all_m_at_tau_abs.append(metrics["m_at_tau_abs"])
        all_m_at_tau_rel.append(metrics["m_at_tau_rel"])
        if metrics["spearman_rho"] is not None:
            all_spearman.append(metrics["spearman_rho"])

        if metrics.get("suff") is not None:
            all_suff.append(metrics["suff"])
        if metrics.get("comp") is not None:
            all_comp.append(metrics["comp"])
            all_comp_ood.append(metrics["comp"])

        if faith_m_values:
            scores = compute_feature_mask_scores(
                explainer=explainer,
                model=model,
                tokens=tokens_batch,
                layer=layer,
                position=eval_position,
                h=activations,
                h_out=activations_out,
                logits_original=logits_original,
                target_token_idx=tgt_idx,
                model_type=model_type,
                m_values=faith_m_values,
                random_repeats=faith_random_repeats,
                empty_mode=faith_empty_mode,
                rank_mode=faith_rank_mode,
                rank_top_f=faith_rank_top_f,
                hook_site_mode=faith_hook_site_mode,
                compute_random_baseline=faith_compute_random_baseline,
                compute_sufficiency=True,
                precomputed_z=metrics.get("faith_z"),
                precomputed_top_idx=metrics.get("faith_top_idx"),
                precomputed_s0=metrics.get("faith_s0"),
            )
            aopc = compute_aopc_and_normalized(scores, faith_m_values)
            comp_suff = compute_comp_suff(scores, faith_m_values)

            all_aopc.append(aopc["aopc"])
            all_n_aopc.append(aopc["n_aopc"])
            if aopc.get("aopc_rand") is not None:
                all_aopc_rand.append(aopc["aopc_rand"])
            if aopc.get("gap_aopc") is not None:
                all_gap_aopc.append(aopc["gap_aopc"])
            if aopc.get("gap_n_aopc") is not None:
                all_gap_n_aopc.append(aopc["gap_n_aopc"])
            all_delta_max.append(aopc["delta_max"])
            all_aopc_negative_frac.append(float(scores.get("aopc_negative_frac", float("nan"))))
            all_topm_logit_increase_frac.append(float(scores.get("topm_logit_increase_frac", float("nan"))))
            if aopc.get("flag_delta_max_nonpos"):
                delta_max_nonpos += 1
            if aopc["delta_max"] > 0:
                all_n_aopc_delta_pos.append(aopc["n_aopc"])
            else:
                all_n_aopc_delta_nonpos.append(aopc["n_aopc"])

            norm_valid = bool(aopc["delta_max"] > norm_min_delta)
            if norm_valid:
                norm_valid_count += 1
                all_n_aopc_norm_valid.append(aopc["n_aopc"])
            else:
                norm_invalid_count += 1
                all_n_aopc_norm_invalid.append(aopc["n_aopc"])

            # Comp/Suff metrics
            if not np.isnan(comp_suff.get("comp_at_m", float("nan"))):
                all_comp_m.append(comp_suff["comp_at_m"])
            if not np.isnan(comp_suff.get("suff_at_m", float("nan"))):
                all_suff_m.append(comp_suff["suff_at_m"])
            all_n_comp_m.append(comp_suff["n_comp"])
            if comp_suff.get("n_suff") is not None:
                all_n_suff_m.append(comp_suff["n_suff"])
            if norm_valid:
                if comp_suff.get("n_comp") is not None:
                    all_n_comp_norm_valid.append(comp_suff["n_comp"])
                if comp_suff.get("n_suff") is not None:
                    all_n_suff_norm_valid.append(comp_suff["n_suff"])
            else:
                if comp_suff.get("n_comp") is not None:
                    all_n_comp_norm_invalid.append(comp_suff["n_comp"])
                if comp_suff.get("n_suff") is not None:
                    all_n_suff_norm_invalid.append(comp_suff["n_suff"])

    # Compute FT if batch_id is provided (same no-padding policy as OOD).
    all_comp_id = []
    if batch_id is not None:
        for tokens_id in tqdm(batch_id, desc="Evaluating ID (FT)", total=len(batch_id)):
            tokens_batch_id = tokens_id.unsqueeze(0).to(device) if tokens_id.dim() == 1 else tokens_id.to(device)

            if input_hook:
                activations_id = extract_hook_activations(
                    model, tokens_batch_id, layer, position=eval_position, hook_name=input_hook
                )
            else:
                activations_id = extract_residual_stream_activations(
                    model, tokens_batch_id, layer, position=eval_position
                )

            activations_id_out = None
            if getattr(explainer, "output_kind", None) == "mlp_out" and output_hook:
                activations_id_out = extract_hook_activations(
                    model, tokens_batch_id, layer, position=eval_position, hook_name=output_hook
                )

            logits_full_id = get_true_logits_from_model(model, tokens_batch_id)
            validate_logits_shape(logits_full_id, expected_vocab_size=model.cfg.d_vocab)
            logits_original_id = logits_full_id[:, eval_position, :]

            if target_mode == 'gold':
                tgt_idx_id = int(tokens_batch_id[0, -1].item())
            else:
                tgt_idx_id = int(logits_original_id[0].argmax(dim=-1).item())

            metrics_id = _compute_single_example_metrics(
                explainer=explainer,
                model=model,
                tokens=tokens_batch_id,
                layer=layer,
                position=eval_position,
                h=activations_id,
                logits_original=logits_original_id,
                target_token_idx=tgt_idx_id,
                model_type=model_type,
                device=device,
                h_out=activations_id_out,
                empty_mode=faith_empty_mode,
                rank_mode=faith_rank_mode,
                rank_top_f=faith_rank_top_f,
                hook_site_mode=faith_hook_site_mode,
            )
            if metrics_id.get("comp") is not None:
                all_comp_id.append(metrics_id["comp"])
    
    # Aggregate
    results: Dict[str, object] = {
        "mean_auc": float(np.mean(all_aucs)) if all_aucs else float("nan"),
        "std_auc": float(np.std(all_aucs)) if all_aucs else float("nan"),
        "aucs": all_aucs,
        "mean_auc_keep": float(np.mean(all_aucs)) if all_aucs else float("nan"),
        "std_auc_keep": float(np.std(all_aucs)) if all_aucs else float("nan"),
        "aucs_keep": all_aucs,
        "mean_auc_del": float(np.mean(all_aucs_del)) if all_aucs_del else None,
        "std_auc_del": float(np.std(all_aucs_del)) if all_aucs_del else None,
        "aucs_del": all_aucs_del,
        "aucs_hidden": all_hidden_aucs,
        "suff_values": all_suff,
        "comp_values": all_comp,
        "cf_values": all_cf,
        "faith_empty_mode": faith_empty_mode,
        "faith_rank_mode": faith_rank_mode,
        "faith_rank_top_f": int(faith_rank_top_f),
        "faith_hook_site_mode": faith_hook_site_mode,
    }
    if all_hidden_aucs:
        results["mean_auc_hidden"] = float(np.mean(all_hidden_aucs))
        results["std_auc_hidden"] = float(np.std(all_hidden_aucs))
    else:
        results["mean_auc_hidden"] = None
        results["std_auc_hidden"] = None

    # m@τ stats (disabled for speed)
    # if all_m_at_tau_abs:
    #     arr_abs = np.array(all_m_at_tau_abs, dtype=np.float64)
    #     results["m_at_tau_abs_mean"] = float(arr_abs.mean())
    #     results["m_at_tau_abs_std"] = float(arr_abs.std())
    # if all_m_at_tau_rel:
    #     arr_rel = np.array(all_m_at_tau_rel, dtype=np.float64)
    #     results["m_at_tau_rel_mean"] = float(arr_rel.mean())
    #     results["m_at_tau_rel_std"] = float(arr_rel.std())

    # Spearman ρ stats (disabled for speed)
    # if all_spearman:
    #     arr_sp = np.array(all_spearman, dtype=np.float64)
    #     results["spearman_rho_mean"] = float(arr_sp.mean())
    #     results["spearman_rho_std"] = float(arr_sp.std())
    # else:
    #     results["spearman_rho_mean"] = None
    #     results["spearman_rho_std"] = None
    #
    # Sufficiency / Comprehensiveness / CF stats
    if all_suff:
        arr_suff = np.array(all_suff, dtype=np.float64)
        results["suff_mean"] = float(arr_suff.mean())
        results["suff_std"] = float(arr_suff.std())
    else:
        results["suff_mean"] = None
        results["suff_std"] = None

    if all_comp:
        arr_comp = np.array(all_comp, dtype=np.float64)
        results["comp_mean"] = float(arr_comp.mean())
        results["comp_std"] = float(arr_comp.std())
    else:
        results["comp_mean"] = None
        results["comp_std"] = None

    if all_cf:
        arr_cf = np.array(all_cf, dtype=np.float64)
        results["cf_mean"] = float(arr_cf.mean())
        results["cf_std"] = float(arr_cf.std())
    else:
        results["cf_mean"] = None
        results["cf_std"] = None

    # AOPC / Comp / Suff over M (optional)
    if all_aopc:
        arr = np.array(all_aopc, dtype=np.float64)
        results["aopc_mean"] = float(arr.mean())
        results["aopc_std"] = float(arr.std())
    else:
        results["aopc_mean"] = None
        results["aopc_std"] = None

    if all_aopc_rand:
        arr = np.array(all_aopc_rand, dtype=np.float64)
        results["aopc_rand_mean"] = float(arr.mean())
        results["aopc_rand_std"] = float(arr.std())
    else:
        results["aopc_rand_mean"] = None
        results["aopc_rand_std"] = None

    if all_gap_aopc:
        arr = np.array(all_gap_aopc, dtype=np.float64)
        results["gap_aopc_mean"] = float(arr.mean())
        results["gap_aopc_std"] = float(arr.std())
    else:
        results["gap_aopc_mean"] = None
        results["gap_aopc_std"] = None

    if all_n_aopc:
        arr = np.array(all_n_aopc, dtype=np.float64)
        results["n_aopc_mean"] = float(arr.mean())
        results["n_aopc_std"] = float(arr.std())
    else:
        results["n_aopc_mean"] = None
        results["n_aopc_std"] = None

    if all_gap_n_aopc:
        arr = np.array(all_gap_n_aopc, dtype=np.float64)
        results["gap_n_aopc_mean"] = float(arr.mean())
        results["gap_n_aopc_std"] = float(arr.std())
    else:
        results["gap_n_aopc_mean"] = None
        results["gap_n_aopc_std"] = None

    if all_comp_m:
        arr = np.array(all_comp_m, dtype=np.float64)
        results["comp_m_mean"] = float(arr.mean())
        results["comp_m_std"] = float(arr.std())
    else:
        results["comp_m_mean"] = None
        results["comp_m_std"] = None

    if all_suff_m:
        arr = np.array(all_suff_m, dtype=np.float64)
        results["suff_m_mean"] = float(arr.mean())
        results["suff_m_std"] = float(arr.std())
    else:
        results["suff_m_mean"] = None
        results["suff_m_std"] = None

    if all_n_comp_m:
        arr = np.array(all_n_comp_m, dtype=np.float64)
        results["n_comp_raw_mean"] = float(arr.mean())
        results["n_comp_raw_std"] = float(arr.std())
        results["n_comp_m_mean"] = results["n_comp_raw_mean"]
        results["n_comp_m_std"] = results["n_comp_raw_std"]
    else:
        results["n_comp_raw_mean"] = None
        results["n_comp_raw_std"] = None
        results["n_comp_m_mean"] = None
        results["n_comp_m_std"] = None

    if all_n_suff_m:
        arr = np.array(all_n_suff_m, dtype=np.float64)
        results["n_suff_raw_mean"] = float(arr.mean())
        results["n_suff_raw_std"] = float(arr.std())
        results["n_suff_m_mean"] = results["n_suff_raw_mean"]
        results["n_suff_m_std"] = results["n_suff_raw_std"]
    else:
        results["n_suff_raw_mean"] = None
        results["n_suff_raw_std"] = None
        results["n_suff_m_mean"] = None
        results["n_suff_m_std"] = None

    if all_comp_at_m:
        arr = np.array(all_comp_at_m, dtype=np.float64)
        results["comp_at_m_mean"] = float(arr.mean())
        results["comp_at_m_std"] = float(arr.std())
    else:
        results["comp_at_m_mean"] = None
        results["comp_at_m_std"] = None

    if all_suff_at_m:
        arr = np.array(all_suff_at_m, dtype=np.float64)
        results["suff_at_m_mean"] = float(arr.mean())
        results["suff_at_m_std"] = float(arr.std())
    else:
        results["suff_at_m_mean"] = None
        results["suff_at_m_std"] = None

    if all_delta_max:
        arr = np.array(all_delta_max, dtype=np.float64)
        results["delta_max_mean"] = float(arr.mean())
        results["delta_max_std"] = float(arr.std())
        results["delta_max_values"] = [float(v) for v in all_delta_max]
        results["delta_max_nonpos_frac"] = float(delta_max_nonpos / max(1, len(all_delta_max)))
    else:
        results["delta_max_mean"] = None
        results["delta_max_std"] = None
        results["delta_max_nonpos_frac"] = None

    if all_recon_gap:
        arr = np.array(all_recon_gap, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        results["recon_gap_mean"] = float(arr.mean()) if arr.size > 0 else None
        results["recon_gap_std"] = float(arr.std()) if arr.size > 0 else None
    else:
        results["recon_gap_mean"] = None
        results["recon_gap_std"] = None

    if all_recon_s_true:
        arr = np.array(all_recon_s_true, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        results["s_true_mean"] = float(arr.mean()) if arr.size > 0 else None
    else:
        results["s_true_mean"] = None

    if all_recon_s_recon:
        arr = np.array(all_recon_s_recon, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        results["s_recon_mean"] = float(arr.mean()) if arr.size > 0 else None
    else:
        results["s_recon_mean"] = None

    if all_aopc_negative_frac:
        arr = np.array(all_aopc_negative_frac, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        results["aopc_negative_frac_mean"] = float(arr.mean()) if arr.size > 0 else None
    else:
        results["aopc_negative_frac_mean"] = None

    if all_topm_logit_increase_frac:
        arr = np.array(all_topm_logit_increase_frac, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        results["topm_logit_increase_frac_mean"] = float(arr.mean()) if arr.size > 0 else None
    else:
        results["topm_logit_increase_frac_mean"] = None

    if all_n_aopc_delta_pos:
        arr = np.array(all_n_aopc_delta_pos, dtype=np.float64)
        results["n_aopc_mean_delta_pos"] = float(arr.mean())
        results["n_aopc_std_delta_pos"] = float(arr.std())
        results["n_delta_pos_samples"] = int(arr.size)
    else:
        results["n_aopc_mean_delta_pos"] = None
        results["n_aopc_std_delta_pos"] = None
        results["n_delta_pos_samples"] = 0

    if all_n_aopc_delta_nonpos:
        arr = np.array(all_n_aopc_delta_nonpos, dtype=np.float64)
        results["n_aopc_mean_delta_nonpos"] = float(arr.mean())
        results["n_aopc_std_delta_nonpos"] = float(arr.std())
        results["n_delta_nonpos_samples"] = int(arr.size)
    else:
        results["n_aopc_mean_delta_nonpos"] = None
        results["n_aopc_std_delta_nonpos"] = None
        results["n_delta_nonpos_samples"] = 0

    if all_comp_m_delta_pos:
        arr = np.array(all_comp_m_delta_pos, dtype=np.float64)
        results["comp_m_mean_delta_pos"] = float(arr.mean())
        results["comp_m_std_delta_pos"] = float(arr.std())
    else:
        results["comp_m_mean_delta_pos"] = None
        results["comp_m_std_delta_pos"] = None

    if all_comp_m_delta_nonpos:
        arr = np.array(all_comp_m_delta_nonpos, dtype=np.float64)
        results["comp_m_mean_delta_nonpos"] = float(arr.mean())
        results["comp_m_std_delta_nonpos"] = float(arr.std())
    else:
        results["comp_m_mean_delta_nonpos"] = None
        results["comp_m_std_delta_nonpos"] = None

    if all_n_aopc_norm_valid:
        arr = np.array(all_n_aopc_norm_valid, dtype=np.float64)
        results["n_aopc_primary_mean"] = float(arr.mean())
        results["n_aopc_primary_std"] = float(arr.std())
    else:
        results["n_aopc_primary_mean"] = results.get("n_aopc_mean")
        results["n_aopc_primary_std"] = results.get("n_aopc_std")

    if all_n_comp_norm_valid:
        arr = np.array(all_n_comp_norm_valid, dtype=np.float64)
        results["n_comp_primary_mean"] = float(arr.mean())
        results["n_comp_primary_std"] = float(arr.std())
    else:
        results["n_comp_primary_mean"] = results.get("n_comp_raw_mean")
        results["n_comp_primary_std"] = results.get("n_comp_raw_std")

    if all_n_suff_norm_valid:
        arr = np.array(all_n_suff_norm_valid, dtype=np.float64)
        results["n_suff_primary_mean"] = float(arr.mean())
        results["n_suff_primary_std"] = float(arr.std())
    else:
        results["n_suff_primary_mean"] = results.get("n_suff_raw_mean")
        results["n_suff_primary_std"] = results.get("n_suff_raw_std")

    results["n_comp_mean"] = results["n_comp_primary_mean"]
    results["n_comp_std"] = results["n_comp_primary_std"]
    results["n_suff_mean"] = results["n_suff_primary_mean"]
    results["n_suff_std"] = results["n_suff_primary_std"]

    results["comp_m_primary_mean"] = (
        results["comp_m_mean_delta_pos"]
        if results["comp_m_mean_delta_pos"] is not None
        else results.get("comp_m_mean")
    )
    results["norm_valid_frac"] = float(norm_valid_count / max(1, norm_valid_count + norm_invalid_count))
    results["norm_invalid_frac"] = float(norm_invalid_count / max(1, norm_valid_count + norm_invalid_count))
    results["faith_norm_min_delta"] = norm_min_delta

    # No right-padding contamination in current per-example evaluation path.
    results["pad_position_frac"] = 0.0

    # Faithfulness Transfer (FT) computation
    if batch_id is not None:
        if len(all_comp_ood) > 0 and len(all_comp_id) > 0:
            # Ensure same length (pad with zeros if needed, or truncate)
            min_len = min(len(all_comp_ood), len(all_comp_id))
            comp_ood_arr = np.array(all_comp_ood[:min_len], dtype=np.float64)
            comp_id_arr = np.array(all_comp_id[:min_len], dtype=np.float64)
            eps = 1e-8
            ft_ratios = comp_ood_arr / (comp_id_arr + eps)
            results["ft_mean"] = float(ft_ratios.mean())
            results["ft_std"] = float(ft_ratios.std())
            print(f"  [FT] Computed from {min_len} pairs: OOD Comp={comp_ood_arr.mean():.3f}, ID Comp={comp_id_arr.mean():.3f}, FT={results['ft_mean']:.3f}")
        else:
            results["ft_mean"] = None
            results["ft_std"] = None
            if batch_id is not None:
                print(f"  [FT] Warning: Cannot compute FT - OOD Comp samples: {len(all_comp_ood)}, ID Comp samples: {len(all_comp_id)}")
    else:
        results["ft_mean"] = None
        results["ft_std"] = None

    keep_keys = {
        "mean_auc", "std_auc", "aucs",
        "mean_auc_keep", "std_auc_keep", "aucs_keep",
        "mean_auc_hidden", "std_auc_hidden", "aucs_hidden",
        "suff_mean", "suff_std", "suff_values",
        "comp_mean", "comp_std", "comp_values",
        "n_suff_raw_mean", "n_suff_raw_std",
        "n_suff_mean", "n_suff_std", "n_suff_primary_mean", "n_suff_primary_std",
        "n_comp_raw_mean", "n_comp_raw_std",
        "n_comp_mean", "n_comp_std", "n_comp_primary_mean", "n_comp_primary_std",
        "n_aopc_mean", "n_aopc_std", "n_aopc_primary_mean", "n_aopc_primary_std",
        "n_aopc_mean_delta_pos", "n_aopc_std_delta_pos",
        "n_aopc_mean_delta_nonpos", "n_aopc_std_delta_nonpos",
        "n_delta_pos_samples", "n_delta_nonpos_samples",
        "delta_max_mean", "delta_max_std", "delta_max_nonpos_frac",
        "norm_valid_frac", "norm_invalid_frac", "faith_norm_min_delta",
        "faith_empty_mode", "faith_rank_mode", "faith_rank_top_f", "faith_hook_site_mode",
        "ft_mean", "ft_std", "pad_position_frac",
    }
    for key in list(results.keys()):
        if key not in keep_keys:
            results.pop(key, None)

    return results

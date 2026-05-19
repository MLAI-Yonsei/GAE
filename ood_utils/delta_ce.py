"""
ΔCE (CE Loss Increase) metric for evaluating SAE/Transcoder faithfulness.

ΔCE = CE(model with SAE/TC reconstruction) - CE(original model)

Lower ΔCE means the reconstruction better preserves the model's output distribution.
"""
from __future__ import annotations

import sys
import os
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import extract_hook_activations, extract_residual_stream_activations
from utils import (
    get_true_logits_from_model,
    forward_with_override_logits,
    validate_logits_shape,
    _resolve_hook_name,
)
from ood_utils.evaluation import (
    _resolve_faith_hook_site,
    _extract_faith_features,
    _resolve_decoder_override,
    _decode_features,
)
from ood_utils.saelens_adapters import EXPLAINER_TYPES
from config import Config


def _ce_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute per-example cross-entropy loss.

    Args:
        logits: [B, V] — logits at a single position
        target_ids: [B] — target token IDs (long)

    Returns:
        ce: [B] — per-example CE loss (nats)
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [B, V]
    ce = -log_probs[torch.arange(logits.shape[0], device=logits.device), target_ids]
    return ce


def compute_delta_ce(
    model: HookedTransformer,
    explainer,
    tokens_list: List[torch.Tensor],
    layer: int,
    model_type: str = "transcoder",
    device: Optional[torch.device] = None,
    target_mode: str = "gold",
    hook_site_mode: str = "auto",
    batch_size: int = 1,
    return_diagnostics: bool = False,
) -> Dict[str, object]:
    """
    Compute ΔCE = mean_CE(reconstructed) - mean_CE(original) over evaluation samples.

    For each input sequence the metric is:
        CE_orig : cross-entropy of the original model at position pos=-1
                  against the next token (gold target, pos=-1 means target is tokens[-1])
        CE_recon: same CE after replacing the hidden activation at (layer, hook_site)
                  with the SAE/TC full reconstruction  z → D z  (using the same decoder
                  resolution logic as the existing faithfulness metrics)
        ΔCE_i   = CE_recon_i - CE_orig_i

    Args:
        model         : HookedTransformer (frozen)
        explainer     : SAE or Transcoder wrapper with the hook-point attributes
        tokens_list   : List of 1-D or 2-D token tensors, one per example.
                        Each tensor should have length ≥ 2 (need at least one context + target).
        layer         : Layer index where the explainer operates
        model_type    : "transcoder" or "sae"
        device        : Computation device (defaults to Config.device)
        target_mode   : "gold"    — next-token CE: target = tokens[-1], context = tokens[:-1]
                        "argmax"  — self-consistency: target = argmax of original logits at pos=-1
        hook_site_mode: How to resolve the hook point (passed to _resolve_faith_hook_site)
        batch_size    : Number of examples per forward pass (currently only 1 is supported
                        because examples may have different sequence lengths; set > 1 only if
                        all sequences have the same length and are already padded)

    Returns:
        dict with keys:
            "mean_delta_ce"   : float — mean ΔCE across examples (primary scalar)
            "std_delta_ce"    : float — std of per-example ΔCE
            "mean_ce_orig"    : float — mean CE of the original model
            "mean_ce_recon"   : float — mean CE of the reconstructed model
            "delta_ce_values" : List[float] — per-example ΔCE
            "ce_orig_values"  : List[float] — per-example CE_orig
            "ce_recon_values" : List[float] — per-example CE_recon
            "n_examples"      : int — number of evaluated examples
            "target_mode"     : str
            "hook_site_mode"  : str
    """
    if device is None:
        device = Config.device

    model.to(device)
    model.eval()
    if isinstance(explainer, EXPLAINER_TYPES):
        explainer.to(device)
        explainer.eval()

    if target_mode not in ("gold", "argmax"):
        raise ValueError(f"Unknown target_mode='{target_mode}'. Use 'gold' or 'argmax'.")

    # Resolve hook site once (same logic as evaluate_causal_faithfulness)
    hook_site = _resolve_faith_hook_site(explainer, hook_site_mode=hook_site_mode)

    input_hook = getattr(explainer, "input_hook_point", None)
    output_hook = getattr(explainer, "output_hook_point", None)

    # For gold-target mode we evaluate at the second-to-last position so the last token
    # is available as the target.  For argmax we evaluate at pos=-1 (same as faithfulness).
    eval_position = -2 if target_mode == "gold" else -1

    delta_ce_values: List[float] = []
    ce_orig_values: List[float] = []
    ce_recon_values: List[float] = []
    kl_values: List[float] = []
    top1_agree: List[float] = []
    top5_agree: List[float] = []
    mse_values: List[float] = []
    h_norm_orig: List[float] = []
    h_norm_recon: List[float] = []

    for tokens in tqdm(tokens_list, desc="ΔCE evaluation", leave=False):
        # Ensure [1, L] shape
        tokens_batch = tokens.unsqueeze(0).to(device) if tokens.dim() == 1 else tokens.to(device)
        B, L = tokens_batch.shape

        if L < 2:
            # Cannot compute gold-mode CE without at least one context token + one target token.
            continue

        # ---- Original model logits ----
        with torch.no_grad():
            logits_full = get_true_logits_from_model(model, tokens_batch)
        validate_logits_shape(logits_full, expected_vocab_size=model.cfg.d_vocab)
        # logits at the evaluation position: [B, V]
        logits_at_pos = logits_full[:, eval_position, :]

        # ---- Target token ID ----
        if target_mode == "gold":
            # Target is the actual last token; context ends at eval_position (-2).
            target_ids = tokens_batch[:, -1]  # [B]
        else:  # argmax
            target_ids = logits_at_pos.argmax(dim=-1)  # [B]
        target_ids = target_ids.to(device=device, dtype=torch.long)

        # ---- Extract hidden activations at eval_position ----
        with torch.no_grad():
            if input_hook:
                h = extract_hook_activations(
                    model, tokens_batch, layer, position=eval_position, hook_name=input_hook
                )
            else:
                h = extract_residual_stream_activations(
                    model, tokens_batch, layer, position=eval_position
                )
            h = h.to(device)  # [B, d_model]

            # For transcoder: also extract output activation (MLP out)
            h_out = None
            if getattr(explainer, "output_kind", None) == "mlp_out" and output_hook:
                h_out = extract_hook_activations(
                    model, tokens_batch, layer, position=eval_position, hook_name=output_hook
                )
                h_out = h_out.to(device)

        # ---- Encode → decode to get reconstruction ----
        with torch.no_grad():
            z = _extract_faith_features(explainer, h, device, h_target=h_out if h_out is not None else h)
            # z: [B, K]

            decoder_override, decoder_bias = _resolve_decoder_override(explainer, device)

            # Reconstruct hidden activation from all features (full reconstruction)
            h_recon = _decode_features(
                explainer=explainer,
                feats=z,
                h_template=h_out if h_out is not None else h,
                decoder_override=decoder_override,
                decoder_bias=decoder_bias,
                empty_mode="zero_resid",
            )
            # h_recon: [B, d_model]


        # ---- Forward pass with reconstruction injected ----
        with torch.no_grad():
            logits_recon_full = forward_with_override_logits(
                model=model,
                input_ids=tokens_batch,
                layer=layer,
                position=eval_position,
                hook_site=hook_site,
                h_override=h_recon,
            )
        validate_logits_shape(logits_recon_full, expected_vocab_size=model.cfg.d_vocab)
        logits_recon_at_pos = logits_recon_full[:, eval_position, :]  # [B, V]

        # ---- Compute CE for each example in the batch ----
        with torch.no_grad():
            ce_orig = _ce_from_logits(logits_at_pos, target_ids)      # [B]
            ce_recon = _ce_from_logits(logits_recon_at_pos, target_ids)  # [B]
            delta_ce = ce_recon - ce_orig                               # [B]

        for b in range(B):
            ce_orig_values.append(float(ce_orig[b].item()))
            ce_recon_values.append(float(ce_recon[b].item()))
            delta_ce_values.append(float(delta_ce[b].item()))

        if return_diagnostics:
            with torch.no_grad():
                lp_o = F.log_softmax(logits_at_pos.float(), dim=-1)
                lp_r = F.log_softmax(logits_recon_at_pos.float(), dim=-1)
                p_o = lp_o.exp()
                kl_per = (p_o * (lp_o - lp_r)).sum(dim=-1)  # [B]
                top1_o = logits_at_pos.argmax(dim=-1)  # [B]
                top1_r = logits_recon_at_pos.argmax(dim=-1)  # [B]
                top5_o_sets = [set(torch.topk(logits_at_pos[i], 5).indices.tolist()) for i in range(B)]
                top5_r_sets = [set(torch.topk(logits_recon_at_pos[i], 5).indices.tolist()) for i in range(B)]

                h_ref = h_out if h_out is not None else h
                mse_per = ((h_ref - h_recon) ** 2).mean(dim=-1)  # [B]
                n_o = h_ref.norm(dim=-1)  # [B]
                n_r = h_recon.norm(dim=-1)  # [B]

            for b in range(B):
                kl_values.append(float(kl_per[b].item()))
                top1_agree.append(1.0 if int(top1_o[b].item()) == int(top1_r[b].item()) else 0.0)
                top5_agree.append(1.0 if int(top1_o[b].item()) in top5_r_sets[b] else 0.0)
                mse_values.append(float(mse_per[b].item()))
                h_norm_orig.append(float(n_o[b].item()))
                h_norm_recon.append(float(n_r[b].item()))

    n = len(delta_ce_values)
    if n == 0:
        return {
            "mean_delta_ce": float("nan"),
            "std_delta_ce": float("nan"),
            "mean_ce_orig": float("nan"),
            "mean_ce_recon": float("nan"),
            "delta_ce_values": [],
            "ce_orig_values": [],
            "ce_recon_values": [],
            "n_examples": 0,
            "target_mode": target_mode,
            "hook_site_mode": hook_site_mode,
        }

    arr = np.array(delta_ce_values, dtype=np.float64)
    arr_orig = np.array(ce_orig_values, dtype=np.float64)
    arr_recon = np.array(ce_recon_values, dtype=np.float64)

    out = {
        "mean_delta_ce": float(arr.mean()),
        "std_delta_ce": float(arr.std()),
        "mean_ce_orig": float(arr_orig.mean()),
        "mean_ce_recon": float(arr_recon.mean()),
        "delta_ce_values": delta_ce_values,
        "ce_orig_values": ce_orig_values,
        "ce_recon_values": ce_recon_values,
        "n_examples": n,
        "target_mode": target_mode,
        "hook_site_mode": hook_site_mode,
    }

    if return_diagnostics and kl_values:
        kl_arr = np.array(kl_values, dtype=np.float64)
        t1 = np.array(top1_agree, dtype=np.float64)
        t5 = np.array(top5_agree, dtype=np.float64)
        mse_arr = np.array(mse_values, dtype=np.float64)
        n_o_arr = np.array(h_norm_orig, dtype=np.float64)
        n_r_arr = np.array(h_norm_recon, dtype=np.float64)
        out["diagnostics"] = {
            "kl_mean": float(kl_arr.mean()),
            "kl_std": float(kl_arr.std()),
            "top1_agree_rate": float(t1.mean()),
            "top5_agree_rate": float(t5.mean()),
            "mse_mean": float(mse_arr.mean()),
            "mse_std": float(mse_arr.std()),
            "h_norm_orig_mean": float(n_o_arr.mean()),
            "h_norm_recon_mean": float(n_r_arr.mean()),
            "h_norm_ratio": float(n_r_arr.mean() / n_o_arr.mean()) if n_o_arr.mean() > 0 else float("nan"),
            "kl_values": kl_values[:200],
            "top1_agree_values": top1_agree[:200],
            "mse_values": mse_values[:200],
        }

    return out


def _capture_full_seq_activations(
    model: HookedTransformer,
    tokens_batch: torch.Tensor,
    hook_name_full: str,
) -> torch.Tensor:
    """Capture [B, L, d_model] activations at a named hook with a single forward pass."""
    captured: Dict[str, torch.Tensor] = {}

    def _capture_hook(act, hook):
        captured["v"] = act.clone()
        return act

    model.run_with_hooks(tokens_batch, fwd_hooks=[(hook_name_full, _capture_hook)])
    return captured["v"]


def compute_delta_ce_full_seq(
    model: HookedTransformer,
    explainer,
    tokens_list: List[torch.Tensor],
    layer: int,
    model_type: str = "transcoder",
    device: Optional[torch.device] = None,
    hook_site_mode: str = "auto",
    return_diagnostics: bool = False,
    mask_first_position: bool = True,
) -> Dict[str, object]:
    """
    Full-sequence |ΔCE|: inject SAE/Transcoder reconstruction at EVERY position and compute
    averaged next-token CE over all valid prediction positions.

    For a sequence of length L, valid prediction positions are p = 0..L-2 (position p predicts
    tokens[p+1]). If mask_first_position=True, p=0 is also skipped (safety against implicit BOS).

    Per-example ΔCE is the MEAN (over valid positions) of CE_recon - CE_orig. The metric target
    is |ΔCE| -> 0 (signed value stored; absolute comparison used for SOTA claims). Matches the
    community-standard protocol used by Gao 2024 (OpenAI), Rajamanoharan 2024 (DeepMind),
    Lieberum 2024 (Gemma Scope), SAEBench 2025.

    All returned keys are prefixed `full_seq_` to coexist with `compute_delta_ce` output in the
    same baseline dict without collision.

    Args:
        model         : HookedTransformer (frozen).
        explainer     : SAE / Transcoder wrapper with hook-point attributes.
        tokens_list   : List of 1-D or 2-D token tensors, one per example (length >= 3 preferred).
        layer         : Layer index where the explainer operates.
        model_type    : "transcoder" | "sae".
        device        : Computation device (defaults to Config.device).
        hook_site_mode: Passed to `_resolve_faith_hook_site`.
        return_diagnostics : If True, also compute KL / top1 / top5 / MSE / h_norm averages.
        mask_first_position : If True, skip p=0 from averaging (default; safety against BOS).

    Returns:
        Dict with keys (all scalars are Python floats):
            full_seq_mean_delta_ce        : mean signed ΔCE across examples
            full_seq_std_delta_ce         : std of per-example mean ΔCE
            full_seq_mean_ce_orig         : mean CE_orig across examples
            full_seq_mean_ce_recon        : mean CE_recon across examples
            full_seq_delta_ce_values      : List[float] per-example mean ΔCE
            full_seq_ce_orig_values       : List[float]
            full_seq_ce_recon_values      : List[float]
            full_seq_n_examples           : int
            full_seq_n_positions_total    : int   (sum of valid positions across examples)
            full_seq_mask_first_position  : bool
            full_seq_hook_site_mode       : str
            full_seq_diagnostics (opt.)   : dict with kl_mean / top1_agree / top5_agree / mse / h_norm_*
    """
    if device is None:
        device = Config.device

    model.to(device)
    model.eval()
    if isinstance(explainer, EXPLAINER_TYPES):
        explainer.to(device)
        explainer.eval()

    hook_site = _resolve_faith_hook_site(explainer, hook_site_mode=hook_site_mode)
    input_hook = getattr(explainer, "input_hook_point", None)
    output_hook = getattr(explainer, "output_hook_point", None)

    # Resolve full hook names for ALL-position activation capture
    input_hook_full = _resolve_hook_name(layer, input_hook) if input_hook else _resolve_hook_name(layer, "resid_post")
    output_hook_full = _resolve_hook_name(layer, output_hook) if output_hook else None

    per_example_delta: List[float] = []
    per_example_ce_orig: List[float] = []
    per_example_ce_recon: List[float] = []
    total_valid_positions = 0

    # Diagnostics accumulators (summed across positions; averaged at the end)
    kl_sum = 0.0
    top1_hit_sum = 0
    top5_hit_sum = 0
    mse_sum = 0.0
    h_norm_orig_sum = 0.0
    h_norm_recon_sum = 0.0
    n_pos_diag = 0

    for tokens in tqdm(tokens_list, desc="|ΔCE| full-seq", leave=False):
        tokens_batch = tokens.unsqueeze(0).to(device) if tokens.dim() == 1 else tokens.to(device)
        B, L = tokens_batch.shape
        if L < 2:
            continue

        first = 1 if mask_first_position else 0
        # valid positions are p in [first, L-2] (each p predicts tokens[p+1])
        if L - 1 <= first:
            continue
        P = L - 1 - first  # number of valid positions

        # ---- Original logits (full sequence) ----
        with torch.no_grad():
            logits_full = get_true_logits_from_model(model, tokens_batch)  # [B, L, V]
        validate_logits_shape(logits_full, expected_vocab_size=model.cfg.d_vocab)

        # ---- Capture activations at ALL positions ----
        with torch.no_grad():
            h_all = _capture_full_seq_activations(model, tokens_batch, input_hook_full).to(device)
            h_out_all = None
            if getattr(explainer, "output_kind", None) == "mlp_out" and output_hook_full:
                h_out_all = _capture_full_seq_activations(model, tokens_batch, output_hook_full).to(device)

        d_model = h_all.shape[-1]

        # ---- Encode -> decode across all positions (flatten batch*positions) ----
        with torch.no_grad():
            h_flat = h_all.reshape(B * L, d_model)
            h_out_flat = h_out_all.reshape(B * L, d_model) if h_out_all is not None else None

            z = _extract_faith_features(
                explainer,
                h_flat,
                device,
                h_target=h_out_flat if h_out_flat is not None else h_flat,
            )

            decoder_override, decoder_bias = _resolve_decoder_override(explainer, device)

            h_recon_flat = _decode_features(
                explainer=explainer,
                feats=z,
                h_template=h_out_flat if h_out_flat is not None else h_flat,
                decoder_override=decoder_override,
                decoder_bias=decoder_bias,
                empty_mode="zero_resid",
            )  # [B*L, d_model]
            h_recon_all = h_recon_flat.reshape(B, L, d_model)

        # ---- Forward with full-sequence override ----
        with torch.no_grad():
            logits_recon_full = forward_with_override_logits(
                model=model,
                input_ids=tokens_batch,
                layer=layer,
                position=None,
                hook_site=hook_site,
                h_override=h_recon_all,
            )
        validate_logits_shape(logits_recon_full, expected_vocab_size=model.cfg.d_vocab)

        # ---- CE over valid positions ----
        with torch.no_grad():
            lp_o = logits_full[:, first : L - 1, :]          # [B, P, V]
            lp_r = logits_recon_full[:, first : L - 1, :]    # [B, P, V]
            tgt = tokens_batch[:, first + 1 : L]             # [B, P]

            ce_o_flat = _ce_from_logits(lp_o.reshape(B * P, -1), tgt.reshape(B * P))
            ce_r_flat = _ce_from_logits(lp_r.reshape(B * P, -1), tgt.reshape(B * P))
            ce_o_mat = ce_o_flat.reshape(B, P)
            ce_r_mat = ce_r_flat.reshape(B, P)

            ce_o_mean = ce_o_mat.mean(dim=1)
            ce_r_mean = ce_r_mat.mean(dim=1)
            d_mean = ce_r_mean - ce_o_mean

        for b in range(B):
            per_example_ce_orig.append(float(ce_o_mean[b].item()))
            per_example_ce_recon.append(float(ce_r_mean[b].item()))
            per_example_delta.append(float(d_mean[b].item()))
        total_valid_positions += B * P

        if return_diagnostics:
            with torch.no_grad():
                lsm_o = F.log_softmax(lp_o.float(), dim=-1)
                lsm_r = F.log_softmax(lp_r.float(), dim=-1)
                kl = (lsm_o.exp() * (lsm_o - lsm_r)).sum(dim=-1)          # [B, P]
                top1_o = lp_o.argmax(dim=-1)                              # [B, P]
                top1_r = lp_r.argmax(dim=-1)                              # [B, P]
                top5_r = torch.topk(lp_r, 5, dim=-1).indices              # [B, P, 5]
                top5_hit = (top5_r == top1_o.unsqueeze(-1)).any(dim=-1)   # [B, P]

                h_ref_vp = (h_out_all if h_out_all is not None else h_all)[:, first : L - 1, :]
                h_rec_vp = h_recon_all[:, first : L - 1, :]
                mse_vp = ((h_ref_vp - h_rec_vp) ** 2).mean(dim=-1)        # [B, P]
                n_o_vp = h_ref_vp.norm(dim=-1)                            # [B, P]
                n_r_vp = h_rec_vp.norm(dim=-1)                            # [B, P]

            kl_sum += float(kl.sum().item())
            top1_hit_sum += int((top1_o == top1_r).sum().item())
            top5_hit_sum += int(top5_hit.sum().item())
            mse_sum += float(mse_vp.sum().item())
            h_norm_orig_sum += float(n_o_vp.sum().item())
            h_norm_recon_sum += float(n_r_vp.sum().item())
            n_pos_diag += B * P

    n = len(per_example_delta)
    if n == 0:
        return {
            "full_seq_mean_delta_ce": float("nan"),
            "full_seq_std_delta_ce": float("nan"),
            "full_seq_mean_ce_orig": float("nan"),
            "full_seq_mean_ce_recon": float("nan"),
            "full_seq_delta_ce_values": [],
            "full_seq_ce_orig_values": [],
            "full_seq_ce_recon_values": [],
            "full_seq_n_examples": 0,
            "full_seq_n_positions_total": 0,
            "full_seq_mask_first_position": mask_first_position,
            "full_seq_hook_site_mode": hook_site_mode,
        }

    arr = np.array(per_example_delta, dtype=np.float64)
    out = {
        "full_seq_mean_delta_ce": float(arr.mean()),
        "full_seq_std_delta_ce": float(arr.std()),
        "full_seq_mean_ce_orig": float(np.mean(per_example_ce_orig)),
        "full_seq_mean_ce_recon": float(np.mean(per_example_ce_recon)),
        "full_seq_delta_ce_values": per_example_delta,
        "full_seq_ce_orig_values": per_example_ce_orig,
        "full_seq_ce_recon_values": per_example_ce_recon,
        "full_seq_n_examples": n,
        "full_seq_n_positions_total": total_valid_positions,
        "full_seq_mask_first_position": mask_first_position,
        "full_seq_hook_site_mode": hook_site_mode,
    }

    if return_diagnostics and n_pos_diag > 0:
        out["full_seq_diagnostics"] = {
            "kl_mean": kl_sum / n_pos_diag,
            "top1_agree_rate": top1_hit_sum / n_pos_diag,
            "top5_agree_rate": top5_hit_sum / n_pos_diag,
            "mse_mean": mse_sum / n_pos_diag,
            "h_norm_orig_mean": h_norm_orig_sum / n_pos_diag,
            "h_norm_recon_mean": h_norm_recon_sum / n_pos_diag,
            "h_norm_ratio": (h_norm_recon_sum / h_norm_orig_sum) if h_norm_orig_sum > 0 else float("nan"),
            "n_positions": n_pos_diag,
        }

    return out

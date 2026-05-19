"""
Diagnostics for Time-shift OOD experiments.

Always-computable checks on OOD eval inputs:
- ReconErr  = ||h - h_hat(all)|| / ||h||
- LogitErr0 = ||logits(h_hat(all)) - logits(h)|| / ||logits(h)||   (baseline drift before ablation)
- Delta-m curve over an m-grid for the self-consistency target token

IMPORTANT:
logits(h_hat) are computed by intervening on the residual stream (TransformerLens hook),
i.e. *true LM logits*, not explainer proxy logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from config import Config
from gae import adapt_features_for_decoder
from ood_utils.saelens_adapters import EXPLAINER_TYPES


@dataclass
class BatchTensors:
    tokens: torch.Tensor  # [B, S]
    lengths: torch.Tensor  # [B]


def _pad_tokens(tokens_list: List[torch.Tensor], device: torch.device) -> BatchTensors:
    """Pad variable-length token sequences with 0s, track true lengths."""
    lengths = torch.tensor([t.shape[0] for t in tokens_list], device=device, dtype=torch.long)
    max_len = int(lengths.max().item()) if len(tokens_list) > 0 else 0
    padded = []
    for t in tokens_list:
        t = t.to(device)
        if t.shape[0] < max_len:
            pad = torch.zeros(max_len - t.shape[0], dtype=t.dtype, device=device)
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    return BatchTensors(tokens=torch.stack(padded, dim=0), lengths=lengths)


def _format_hook_name(layer: int, hook_name: str) -> str:
    if "blocks." in hook_name:
        return hook_name
    if "." not in hook_name and not hook_name.startswith("hook_"):
        hook_name = f"hook_{hook_name}"
    return f"blocks.{layer}.{hook_name}"


def _resid_post_hook_name(layer: int) -> str:
    return _format_hook_name(layer, "resid_post")


@torch.no_grad()
def _extract_resid_post_last_tokens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    layer: int,
    hook_name: str = "resid_post",
) -> torch.Tensor:
    """Extract resid_post at the true last-token position per sample. Returns H: [B, d_model]."""
    model.eval()
    device = tokens.device
    batch = tokens.shape[0]
    pos = (lengths - 1).to(device)  # [B]
    out: Optional[torch.Tensor] = None

    def hook_fn(value, hook):
        nonlocal out
        idx = torch.arange(batch, device=device)
        out = value[idx, pos, :].clone()
        return value

    model.run_with_hooks(tokens, fwd_hooks=[(_format_hook_name(layer, hook_name), hook_fn)])
    assert out is not None
    return out


@torch.no_grad()
def _logits_at_last_tokens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute logits at the true last token position for each sample. [B, V]"""
    model.eval()
    logits = model(tokens)  # [B, S, V]
    device = tokens.device
    batch = tokens.shape[0]
    pos = (lengths - 1).to(device)
    idx = torch.arange(batch, device=device)
    return logits[idx, pos, :].contiguous()


@torch.no_grad()
def _logits_with_resid_replacement(
    model: HookedTransformer,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    layer: int,
    resid_replacement: torch.Tensor,  # [B, d]
    hook_name: str = "resid_post",
) -> torch.Tensor:
    """Run model with resid_post(last token) replaced. Return logits at last token [B,V]."""
    model.eval()
    device = tokens.device
    batch = tokens.shape[0]
    pos = (lengths - 1).to(device)
    idx = torch.arange(batch, device=device)

    def hook_fn(value, hook):
        value = value.clone()
        value[idx, pos, :] = resid_replacement.to(device)
        return value

    logits = model.run_with_hooks(tokens, fwd_hooks=[(_format_hook_name(layer, hook_name), hook_fn)])  # [B,S,V]
    return logits[idx, pos, :].contiguous()


@torch.no_grad()
def _features_from_explainer(
    explainer,
    h: torch.Tensor,  # [B,d]
    model_type: str,
    h_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Feature activations z: [B,k]."""
    gae_apply_to = getattr(explainer, "gae_apply_to", None)
    if gae_apply_to == "encoder" and hasattr(explainer, "B_gae"):
        B_gae = explainer.B_gae.to(h.device)  # [d,k]
        return F.relu(h @ B_gae)

    if hasattr(explainer, "get_features"):
        return explainer.get_features(h)
    _, z = explainer(h, return_z=True)
    if gae_apply_to == "decoder" and hasattr(explainer, "B_gae"):
        adapter_mode = getattr(explainer, "gae_feature_adapter", None)
        if adapter_mode:
            return adapt_features_for_decoder(
                z0=z,
                h=h_target if h_target is not None else h,
                decoder=explainer.B_gae.to(h.device),
                adapter_mode=adapter_mode,
                gains=getattr(explainer, "gae_feature_gains", None),
                ridge=float(getattr(explainer, "gae_support_refit_ridge", 1e-3)),
                max_support=int(getattr(explainer, "gae_support_refit_topk", 128)),
            )
    return z


@torch.no_grad()
def _reconstruct_h_from_all_features(
    explainer,
    h: torch.Tensor,  # [B,d]
    z: torch.Tensor,  # [B,k]
    model_type: str,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Return h_hat_all and (for transcoder) empirical B_emp used for reconstruction.
    - SAE: uses decoder(z)
    - Transcoder: computes B_emp via least squares so that h ≈ z @ B_emp^T
    """
    decoder_override = getattr(explainer, "gae_decoder_override", None)
    if decoder_override is not None:
        D = decoder_override.to(h.device)
        h_hat_all = z @ D.T
        decoder_bias = getattr(explainer, "gae_decoder_bias", None)
        if decoder_bias is not None:
            h_hat_all = h_hat_all + decoder_bias.to(h.device)
        return h_hat_all, D

    gae_apply_to = getattr(explainer, "gae_apply_to", None)
    if gae_apply_to == "decoder" and hasattr(explainer, "B_gae"):
        D = explainer.B_gae.to(h.device)
        h_hat_all = z @ D.T
        decoder_bias = getattr(explainer, "gae_decoder_bias", None)
        if decoder_bias is not None:
            h_hat_all = h_hat_all + decoder_bias.to(h.device)
        return h_hat_all, D

    # If a calibrated decoder is available (GAE-specific or generic), use it directly
    if hasattr(explainer, "D_gae"):
        D = explainer.D_gae.to(h.device)  # [d,k]
        h_hat_all = z @ D.T  # [B,d]
        return h_hat_all, D

    if model_type == "sae":
        if hasattr(explainer, "decoder"):
            return explainer.decoder(z), None
        h_hat, _ = explainer(h, return_z=True)
        return h_hat, None

    if model_type != "transcoder":
        raise ValueError(f"Unknown model_type: {model_type}")

    H = h
    Z = z
    b, k = Z.shape
    device = H.device

    method = getattr(Config, "DIAG_RECON_METHOD", "auto")
    if method not in ("normal", "row", "auto"):
        raise ValueError(f"Unknown DIAG_RECON_METHOD='{method}'. Use 'normal', 'row', or 'auto'.")
    if method == "row":
        use_row = True
    elif method == "normal":
        use_row = False
    else:
        use_row = b < k

    if use_row:
        # Row-space solve: avoid k x k matrix
        ZZt = Z @ Z.T + eps * torch.eye(b, device=device, dtype=H.dtype)
        Y = torch.linalg.solve(ZZt, H)  # [B,d]
        B_emp = Y.T @ Z  # [d,k]
    else:
        ZTZ = Z.T @ Z + eps * torch.eye(k, device=device, dtype=H.dtype)
        B_emp = (H.T @ Z) @ torch.linalg.inv(ZTZ)  # [d,k]

    h_hat_all = Z @ B_emp.T  # [B,d]
    return h_hat_all, B_emp


@torch.no_grad()
def compute_ood_diagnostics(
    model: HookedTransformer,
    explainer,
    tokens_list: List[torch.Tensor],
    layer: int,
    model_type: str,
    input_hook_point: Optional[str] = None,
    output_hook_point: Optional[str] = None,
    m_values: Optional[List[int]] = None,
    batch_size: int = 16,
    max_samples: int = 256,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Compute ReconErr / LogitErr0 / delta-m curve on OOD eval inputs."""
    device = Config.device
    model.to(device)
    if isinstance(explainer, EXPLAINER_TYPES):
        explainer.to(device)
        explainer.eval()

    if max_samples is not None and max_samples > 0:
        tokens_list = tokens_list[:max_samples]

    # default: same grid spirit as your AUC code (0..200 step10)
    if m_values is None:
        # Use explainer's k if available; fallback to Config.DICT_SIZE_K for the model.
        k_guess = getattr(explainer, "k", None)
        if k_guess is None:
            k_guess = Config.get_dict_size_k(model.cfg.model_name)
        m_values = list(range(0, min(int(k_guess), 200), 10))
        if 0 not in m_values:
            m_values = [0] + m_values

    recon_errs: List[float] = []
    logit_err0s: List[float] = []
    deltas_by_m: List[List[float]] = [[] for _ in m_values]

    for i in range(0, len(tokens_list), batch_size):
        batch_tokens_list = tokens_list[i : i + batch_size]
        bt = _pad_tokens(batch_tokens_list, device=device)

        if input_hook_point:
            H = _extract_resid_post_last_tokens(
                model, bt.tokens, bt.lengths, layer, hook_name=input_hook_point
            )
        else:
            H = _extract_resid_post_last_tokens(model, bt.tokens, bt.lengths, layer)  # [B,d]

        H_out = H
        if output_hook_point and getattr(explainer, "output_kind", None) == "mlp_out":
            H_out = _extract_resid_post_last_tokens(
                model, bt.tokens, bt.lengths, layer, hook_name=output_hook_point
            )
        logits_orig = _logits_at_last_tokens(model, bt.tokens, bt.lengths)  # [B,V]
        target_idx = logits_orig.argmax(dim=-1)  # [B]

        z_all = _features_from_explainer(
            explainer,
            H,
            model_type=model_type,
            h_target=H_out,
        )  # [B,k]
        k = z_all.shape[1]

        # all-features reconstruction
        h_hat_all, B_emp = _reconstruct_h_from_all_features(explainer, H_out, z_all, model_type=model_type)

        # ReconErr
        recon = (torch.norm(H_out - h_hat_all, dim=-1) / torch.norm(H_out, dim=-1).clamp_min(eps)).detach().cpu().tolist()
        recon_errs.extend(recon)

        # LogitErr0
        hook_for_replacement = output_hook_point if output_hook_point and getattr(explainer, "output_kind", None) == "mlp_out" else "resid_post"
        logits_hat_all = _logits_with_resid_replacement(
            model, bt.tokens, bt.lengths, layer, h_hat_all, hook_name=hook_for_replacement
        )
        log0 = (
            torch.norm(logits_hat_all - logits_orig, dim=-1)
            / torch.norm(logits_orig, dim=-1).clamp_min(eps)
        ).detach().cpu().tolist()
        logit_err0s.extend(log0)

        # Delta-m curve
        gae_apply_to = getattr(explainer, "gae_apply_to", None)
        top_idx = torch.argsort(z_all, dim=1, descending=True)  # [B,k]
        rows = torch.arange(z_all.shape[0], device=device).unsqueeze(1)

        for mi, m in enumerate(m_values):
            m_eff = int(min(max(int(m), 0), k))
            if m_eff == 0:
                z_ret = torch.zeros_like(z_all)
            elif m_eff >= k:
                z_ret = z_all
            else:
                z_ret = torch.zeros_like(z_all)
                sel = top_idx[:, :m_eff]
                z_ret[rows, sel] = z_all[rows, sel]

            decoder_override = getattr(explainer, "gae_decoder_override", None)
            if decoder_override is not None:
                h_hat_m = z_ret @ decoder_override.to(device).T
                decoder_bias = getattr(explainer, "gae_decoder_bias", None)
                if decoder_bias is not None:
                    h_hat_m = h_hat_m + decoder_bias.to(device)
            elif gae_apply_to == "decoder" and hasattr(explainer, "B_gae"):
                h_hat_m = z_ret @ explainer.B_gae.to(device).T
                decoder_bias = getattr(explainer, "gae_decoder_bias", None)
                if decoder_bias is not None:
                    h_hat_m = h_hat_m + decoder_bias.to(device)
            elif hasattr(explainer, "D_gae"):
                h_hat_m = z_ret @ explainer.D_gae.to(device).T
            elif model_type == "sae" or getattr(explainer, "output_kind", None) == "mlp_out":
                h_hat_m = explainer.decoder(z_ret)
            else:
                assert B_emp is not None
                h_hat_m = z_ret @ B_emp.T

            logits_hat_m = _logits_with_resid_replacement(
                model, bt.tokens, bt.lengths, layer, h_hat_m, hook_name=hook_for_replacement
            )
            bidx = torch.arange(logits_orig.shape[0], device=device)
            tgt_orig = logits_orig[bidx, target_idx]
            tgt_hat = logits_hat_m[bidx, target_idx]
            delta = (tgt_orig - tgt_hat).detach().cpu().tolist()
            deltas_by_m[mi].extend(delta)

    recon_np = np.array(recon_errs, dtype=np.float64)
    log0_np = np.array(logit_err0s, dtype=np.float64)
    delta_mean = [float(np.mean(d)) if len(d) else float("nan") for d in deltas_by_m]
    delta_std = [float(np.std(d)) if len(d) else float("nan") for d in deltas_by_m]

    return {
        "n_samples": int(len(tokens_list)),
        "m_values": list(map(int, m_values)),
        "recon_err_mean": float(np.mean(recon_np)) if recon_np.size else float("nan"),
        "recon_err_std": float(np.std(recon_np)) if recon_np.size else float("nan"),
        "logit_err0_mean": float(np.mean(log0_np)) if log0_np.size else float("nan"),
        "logit_err0_std": float(np.std(log0_np)) if log0_np.size else float("nan"),
        "delta_curve_mean": delta_mean,
        "delta_curve_std": delta_std,
    }

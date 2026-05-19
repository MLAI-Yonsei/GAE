import torch
import numpy as np
import os
import json
import argparse
import copy
import math
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

from transformer_lens import HookedTransformer
from model_utils import load_model, set_seed, get_layer
from activation_utils import (
    extract_residual_stream_activations,
    extract_logits,
    extract_mlp_in_out_activations,
    collect_activations_batch,
    collect_hook_activations_batch,
    collect_mlp_in_out_activations_batch,
)
from ood_utils.saelens_adapters import (
    EXPLAINER_TYPES,
    SaeLensSparseAutoencoderAdapter,
    SaeLensTranscoderAdapter,
)
from ood_utils import data_adv, data_domain, data_timeshift
from ood_utils.data_adv import (
    load_ood_dataset,
    load_id_dataset,
    sample_and_tokenize_dataset,
    sample_and_tokenize_dataset_by_token_budget,
    create_factual_prompts,
    create_adversarial_prompts,
    batch_tokenize,
)
from ood_utils.data_utils import _extract_text
from ood_utils.evaluation import (
    evaluate_causal_faithfulness,
    compute_feature_mask_scores,
    compute_aopc_and_normalized,
    compute_comp_suff,
    FAITH_HOOK_SITE_MODES,
)
from ood_utils.diagnostics import compute_ood_diagnostics
from ood_utils.delta_ce import compute_delta_ce, compute_delta_ce_full_seq
from train_explainers import collect_id_activations
from gae import (
    extract_dictionary_from_explainer,
    gae_torch,
    consistent_rotation_decoder,
    decoder_only_refit,
    fit_affine_constrained_decoder,
    fit_diag_feature_gains,
    select_rank_for_gae,
    select_decoder_mix_alpha,
)
from activation_cache import load_activations, save_activations
from config import Config
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.activations_store import ActivationsStore
from sae_training.sparse_autoencoder import SparseAutoencoder as SaeLensSparseAutoencoder
from sae_training.train_sae_on_language_model import train_sae_on_language_model
from training_objectives import erm_loss, term_loss
from transformers import get_scheduler
from saeboost import SAEBoostExplainerAdapter, SAEBoostTranscoderAdapter

# Set HuggingFace and datasets cache directories
os.environ['HF_HOME'] = Config.HF_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = Config.HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = Config.DATASETS_CACHE_DIR
os.makedirs(Config.HF_CACHE_DIR, exist_ok=True)
os.makedirs(Config.DATASETS_CACHE_DIR, exist_ok=True)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _inspect_checkpoint_type(checkpoint_path):
    """
    Inspect checkpoint and return (kind, output_kind) if detectable.
    kind: 'transcoder' or 'sae' or None
    output_kind: 'logits' or 'mlp_out' or None
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and 'cfg' in checkpoint and 'state_dict' in checkpoint:
        cfg = checkpoint['cfg']
        kind = "transcoder" if getattr(cfg, "is_transcoder", False) else "sae"
        output_kind = None
        d_out = getattr(cfg, "d_out", None)
        if kind == "transcoder" and d_out is not None:
            output_kind = "mlp_out" if d_out == getattr(cfg, "d_in", None) else "logits"
        return kind, output_kind
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        config = checkpoint.get('config', {})
        if config.get('transcoder_target') == 'mlp_out':
            return "transcoder", "mlp_out"
        if config.get('transcoder_target') == 'logits':
            return "transcoder", "logits"
    return None, None


def _load_domain_ood_jsonl_records(jsonl_path: Path):
    records = []
    if not jsonl_path.exists():
        return records
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ctx = obj.get("context_ids")
            tgt = obj.get("target_next_token_id")
            if tgt is None:
                tgt = obj.get("original_next_token_id")
            if not isinstance(ctx, list) or tgt is None:
                continue
            # Ensure ints
            try:
                ctx = [int(x) for x in ctx]
                tgt = int(tgt)
            except Exception:
                continue
            obj["context_ids"] = ctx
            obj["target_next_token_id"] = tgt
            records.append(obj)
    return records


def _records_to_tokens(records):
    tokens_list = []
    target_token_ids = []
    for r in records:
        ctx = r.get("context_ids")
        tgt = r.get("target_next_token_id")
        if tgt is None:
            tgt = r.get("original_next_token_id")
        if not isinstance(ctx, list) or tgt is None:
            continue
        tokens_list.append(torch.tensor(ctx, dtype=torch.long))
        target_token_ids.append(int(tgt))
    return tokens_list, target_token_ids


def _records_to_token_dataset(records):
    dataset = []
    for r in records:
        ctx = r.get("context_ids")
        if not isinstance(ctx, list) or len(ctx) == 0:
            continue
        dataset.append({"tokens": ctx})
    return dataset


def load_explainer(checkpoint_path, model_type, d_model, vocab_size=None):
    """
    Load explainer from checkpoint (SAELens-only).

    Args:
        checkpoint_path: Path to checkpoint
        model_type: 'transcoder' or 'sae'
        d_model: Hidden dimension
        vocab_size: Vocabulary size (for Transcoder output-kind inference)

    Returns:
        explainer: Loaded SAELens adapter
    """
    # Use weights_only=False for PyTorch 2.6+ compatibility (checkpoints may contain numpy objects)
    checkpoint = torch.load(checkpoint_path, map_location=Config.device, weights_only=False)

    if not (isinstance(checkpoint, dict) and 'cfg' in checkpoint and 'state_dict' in checkpoint):
        raise ValueError(
            "Only SAELens checkpoints are supported. "
            "Expected a dict with keys {'cfg', 'state_dict'}."
        )

    cfg = checkpoint['cfg']
    state_dict = checkpoint['state_dict']

    # Force SAELens model to current device (avoid CUDA init when unavailable)
    cfg.device = str(Config.device)
    saelens_model = SaeLensSparseAutoencoder(cfg)
    saelens_model.load_state_dict(state_dict)
    saelens_model.to(Config.device)
    saelens_model.eval()

    if getattr(cfg, "is_transcoder", False):
        if model_type != "transcoder":
            raise ValueError(
                f"Checkpoint is SAELens transcoder but model_type='{model_type}'. "
                "Use --explainer transcoder."
            )
        adapter = SaeLensTranscoderAdapter(saelens_model)
        if getattr(cfg, "d_out", None) is not None:
            if vocab_size is not None and cfg.d_out == vocab_size:
                adapter.output_kind = "logits"
            elif cfg.d_out == d_model:
                adapter.output_kind = "mlp_out"
        if getattr(cfg, "d_out", None) is not None and vocab_size is not None and cfg.d_out != vocab_size:
            print(
                f"Info: SAELens transcoder d_out={cfg.d_out} != vocab_size={vocab_size}. "
                "Using hidden-state override with TRUE logits (not logits-space evaluation)."
            )
            print(
                "Info: Overriding internal activations and scoring with final model logits "
                "to avoid d_out/vocab mismatch."
            )
        return adapter

    if model_type != "sae":
        raise ValueError(
            f"Checkpoint is SAELens SAE but model_type='{model_type}'. "
            "Use --explainer sae."
        )
    return SaeLensSparseAutoencoderAdapter(saelens_model)


def _get_explainer_hooks(explainer):
    return (
        getattr(explainer, "input_hook_point", None),
        getattr(explainer, "output_hook_point", None),
    )


def _diagnostics_kwargs(explainer):
    input_hook, output_hook = _get_explainer_hooks(explainer)
    return {"input_hook_point": input_hook, "output_hook_point": output_hook}


def _resolve_gae_geometry_activations(
    explainer,
    model_type: str,
    dict_space: str,
    id_activations: torch.Tensor,
    ood_activations: torch.Tensor,
    id_activations_out: Optional[torch.Tensor] = None,
    ood_activations_out: Optional[torch.Tensor] = None,
):
    """
    Select the activation space used for GAE covariance/rank estimation.

    For decoder-space transcoders, geometry must be estimated in the same output
    space as the decoder dictionary (typically MLP-out), not the encoder input space.
    """
    effective_dict_space = "decoder" if dict_space == "auto" else dict_space
    use_output_space = (
        model_type == "transcoder"
        and effective_dict_space == "decoder"
        and getattr(explainer, "output_kind", None) == "mlp_out"
    )

    if use_output_space:
        if id_activations_out is None or ood_activations_out is None:
            raise ValueError(
                "GAE decoder-space transcoder adaptation requires both ID and OOD output-space activations. "
                "Expected id_activations_out and ood_activations_out."
            )
        return {
            "id": id_activations_out,
            "ood": ood_activations_out,
            "space": "output",
            "source": "hook_output",
            "dict_space": effective_dict_space,
        }

    return {
        "id": id_activations,
        "ood": ood_activations,
        "space": "input",
        "source": "hook_input" if model_type == "transcoder" else "resid_post",
        "dict_space": effective_dict_space,
    }


def _build_results_summary(
    task_name: str,
    model_name: str,
    explainer_type: str,
    ood_set: str,
    layer: int,
    baselines,
    explainer,
    config_extra: Optional[Dict[str, object]] = None,
):
    summary = {
        "task": task_name,
        "model": model_name,
        "explainer": explainer_type,
        "ood_set": ood_set,
        "layer": layer,
        "baselines": {},
        "config": {
            "baselines_requested": list(baselines),
            "output_kind": getattr(explainer, "output_kind", None),
            "input_hook_point": getattr(explainer, "input_hook_point", None),
            "output_hook_point": getattr(explainer, "output_hook_point", None),
        },
    }
    if config_extra:
        summary["config"].update(config_extra)
    return summary


def _activation_cache_metadata_for_explainer(
    explainer,
    *,
    dataset_role: str,
    task_name: str,
    position: int = -1,
    ood_set: Optional[str] = None,
    timeshift_span_mode: Optional[str] = None,
    use_jsonl: Optional[bool] = None,
):
    metadata = {
        "dataset_role": dataset_role,
        "task_name": task_name,
        "position": int(position),
        "explainer_output_kind": getattr(explainer, "output_kind", None),
    }
    if ood_set is not None:
        metadata["ood_set"] = ood_set
    if timeshift_span_mode is not None:
        metadata["timeshift_span_mode"] = timeshift_span_mode
    if use_jsonl is not None:
        metadata["domain_use_jsonl"] = bool(use_jsonl)

    if getattr(explainer, "output_kind", None) == "mlp_out":
        metadata.update({
            "extraction_family": "mlp_in_out",
            "activation_kind": "hook_input",
            "target_kind": "hook_output",
            "input_hook_point": getattr(explainer, "input_hook_point", "ln2.hook_normalized"),
            "output_hook_point": getattr(explainer, "output_hook_point", "hook_mlp_out"),
        })
    else:
        # SAE branch: respect explainer.input_hook_point so cache distinguishes
        # different hook points (e.g. ln2.hook_normalized vs resid_post).
        # Re-fix of B2 (docs/sae_failure_v1.md): originally hardcoded to
        # "resid_post" which silently fed the SAE the wrong activation space.
        sae_input_hook = getattr(explainer, "input_hook_point", None) or "resid_post"
        metadata.update({
            "extraction_family": "hook_input+logits",
            "activation_kind": sae_input_hook,
            "target_kind": "logits",
            "input_hook_point": sae_input_hook,
            "output_hook_point": "logits",
        })
    return metadata


def _activation_cache_type(base_type: str, explainer) -> str:
    if getattr(explainer, "output_kind", None) == "mlp_out":
        return f"{base_type}_mlp_out"
    return base_type


def _get_explainer_dictionary(
    explainer,
    activations: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    """
    Return explainer dictionary B with shape [d_model, k] for geometric metrics.
    Preference order:
      1) GAE-adapted dictionary (B_gae) if present.
      2) Decoder weights (W_dec / decoder.weight) when available.
      3) Empirical dictionary extracted from activations.
    """
    if hasattr(explainer, "use_gae") and getattr(explainer, "use_gae", False) and hasattr(explainer, "B_gae"):
        return explainer.B_gae.detach()

    # SAELens adapters expose saelens.W_dec
    if hasattr(explainer, "saelens") and hasattr(explainer.saelens, "W_dec"):
        return explainer.saelens.W_dec.T.detach()

    # Generic SAE/Transcoder decoders
    if hasattr(explainer, "W_dec"):
        return explainer.W_dec.T.detach()
    if hasattr(explainer, "decoder"):
        dec = explainer.decoder
        if hasattr(dec, "weight"):
            return dec.weight.T.detach()

    # Transcoder-specific fallback: W_tilde if it matches hidden size
    if hasattr(explainer, "W_tilde"):
        B = explainer.W_tilde
        d = activations.shape[1]
        if B.shape[0] == d:
            return B.detach()
        if B.shape[1] == d:
            return B.T.detach()

    # Empirical extraction (uses encoder + least squares)
    return extract_dictionary_from_explainer(explainer, activations, model_type)




def compute_raer(
    explainer,
    id_activations: torch.Tensor,
    ood_activations: torch.Tensor,
    model_type: str,
    r: int = None,
) -> dict:
    """
    RAER (Representation Activation Energy Ratio):
    E_{x~OOD}[||(I - U_r U_r^T) h(x)||^2] / E_{x~OOD}[||h(x)||^2]
    where U_r are top-r eigenvectors of Cov_ID(h).
    """
    with torch.no_grad():
        h_id = id_activations
        h_ood = ood_activations.to(h_id.device)

    # Use float32 for stable covariance/eigendecomposition
    h_id = h_id.to(dtype=torch.float32)
    h_ood = h_ood.to(dtype=torch.float32)

    d_model = h_id.shape[1]
    if r is None:
        r = min(d_model // 4, 64)
    r = max(1, min(r, d_model))

    h_id_centered = h_id - h_id.mean(dim=0, keepdim=True)
    cov_id = (h_id_centered.T @ h_id_centered) / max(1, h_id_centered.shape[0] - 1)
    eigenvals, eigenvecs = torch.linalg.eigh(cov_id)  # ascending
    U_r = eigenvecs[:, -r:]  # top-r
    proj = U_r @ U_r.T

    h_ood_centered = h_ood - h_ood.mean(dim=0, keepdim=True)
    projected_out = h_ood_centered - (h_ood_centered @ proj)

    energy_outside = (projected_out ** 2).sum(dim=1).mean()
    total_energy = (h_ood_centered ** 2).sum(dim=1).mean()
    rare_energy_ratio = energy_outside / (total_energy + 1e-10)

    return {
        "rare_energy_ratio": float(rare_energy_ratio.item()),
        "energy_outside": float(energy_outside.item()),
        "total_energy": float(total_energy.item()),
        "r": int(r),
        "top_r_eigenvals": eigenvals[-r:].flip(0).detach().cpu().tolist(),
    }


def compute_hidden_space_alignment(
    explainer,
    ood_activations: torch.Tensor,
    model_type: str,
    r: int = None,
) -> dict:
    """
    Angle-based subspace alignment between explainer directions and OOD-active subspace.
    Align(U_exp, U_ood) = (1/r) * ||U_exp^T U_ood||_F^2
    where U_exp are the top-r left singular vectors of the explainer dictionary B,
    and U_ood are top-r eigenvectors of Cov_OOD(h).
    """
    with torch.no_grad():
        h = ood_activations

    h = h.to(dtype=torch.float32)

    d_model = h.shape[1]
    if r is None:
        r = min(d_model // 4, 64)
    r = max(1, min(r, d_model))

    # Explainer subspace from dictionary B
    B = _get_explainer_dictionary(explainer, h, model_type).to(h.device, dtype=torch.float32)  # [d_model, k]
    if B.shape[0] != d_model:
        raise ValueError(f"Dictionary B has incompatible shape {B.shape} for d_model={d_model}")

    # SVD for U_exp (top-r left singular vectors)
    U, _, _ = torch.linalg.svd(B, full_matrices=False)
    U_exp = U[:, :r]  # [d_model, r]

    # OOD-active subspace from covariance of hidden activations
    h_centered = h - h.mean(dim=0, keepdim=True)
    cov_ood = (h_centered.T @ h_centered) / max(1, h_centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov_ood)  # ascending
    Q_ood = eigvecs[:, -r:]  # [d_model, r]

    # Angle-based alignment
    alignment_score = float((torch.norm(U_exp.T @ Q_ood, p="fro") ** 2 / r).clamp(0.0, 1.0).item())

    return {
        "hidden_space_alignment": alignment_score,
        "q_ood_rank": int(r),
    }


def compute_energy_weighted_alignment(
    id_activations: torch.Tensor,
    ood_activations: torch.Tensor,
    r: Optional[int] = None,
) -> dict:
    """
    Energy-weighted alignment:
    tr(U_id^T Sigma_ood U_id) / tr(Sigma_ood),
    where U_id are top-r eigenvectors of Cov_ID.
    """
    with torch.no_grad():
        h_id = id_activations.to(dtype=torch.float32)
        h_ood = ood_activations.to(h_id.device, dtype=torch.float32)

    d_model = h_id.shape[1]
    if r is None:
        r = min(d_model // 4, 64)
    r = max(1, min(r, d_model))

    h_id_centered = h_id - h_id.mean(dim=0, keepdim=True)
    cov_id = (h_id_centered.T @ h_id_centered) / max(1, h_id_centered.shape[0] - 1)
    _, eigvecs_id = torch.linalg.eigh(cov_id)
    U_id = eigvecs_id[:, -r:]

    h_ood_centered = h_ood - h_ood.mean(dim=0, keepdim=True)
    sigma_ood = (h_ood_centered.T @ h_ood_centered) / max(1, h_ood_centered.shape[0] - 1)

    numerator = torch.trace(U_id.T @ sigma_ood @ U_id)
    denominator = torch.trace(sigma_ood) + 1e-10
    score = float((numerator / denominator).clamp(0.0, 1.0).item())
    return {
        "energy_weighted_alignment": score,
        "r": int(r),
    }


def _bootstrap_mean_ci(values: List[float], n_bootstrap: int, ci_level: float, seed: int) -> Optional[dict]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = []
    n = arr.size
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        means.append(float(arr[idx].mean()))
    alpha = (100.0 - float(ci_level)) / 2.0
    lower = float(np.percentile(means, alpha))
    upper = float(np.percentile(means, 100.0 - alpha))
    return {
        "mean": float(arr.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": float(ci_level),
        "n_bootstrap": int(n_bootstrap),
        "n_samples": int(n),
    }


def _add_faithfulness_bootstrap_ci(
    res: dict,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> None:
    """Attach bootstrap CIs for per-example faithfulness metrics in res."""
    if n_bootstrap <= 0:
        return
    ci_results = {}
    for key in ["suff_values", "comp_values", "cf_values", "aucs"]:
        vals = res.get(key)
        if isinstance(vals, list) and len(vals) > 0:
            ci = _bootstrap_mean_ci(vals, n_bootstrap, ci_level, seed)
            if ci is not None:
                ci_results[f"{key.replace('_values', '').replace('aucs', 'auc')}_ci"] = ci
    res.update(ci_results)


def _build_halueval_label_subsets(eval_metadata: List[dict]) -> Dict[str, List[int]]:
    hallucinated: List[int] = []
    non_hallucinated: List[int] = []
    for i, meta in enumerate(eval_metadata):
        if not isinstance(meta, dict):
            continue
        label = meta.get("hallucination_label")
        if label is True:
            hallucinated.append(i)
        elif label is False:
            non_hallucinated.append(i)
    return {
        "hallucinated_only": hallucinated,
        "non_hallucinated_only": non_hallucinated,
    }


def _add_behavior_level_bootstrap_ci(
    res: dict,
    eval_metadata: List[dict],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> None:
    """Attach per-behavior-subset bootstrap CIs (e.g. hallucinated vs non-hallucinated)."""
    if n_bootstrap <= 0 or not eval_metadata:
        return
    subsets = _build_halueval_label_subsets(eval_metadata)
    for subset_name, indices in subsets.items():
        if not indices:
            continue
        subset_ci = {}
        for key in ["suff_values", "comp_values", "aucs"]:
            vals = res.get(key)
            if isinstance(vals, list) and len(vals) > 0:
                sub_vals = [vals[i] for i in indices if i < len(vals)]
                ci = _bootstrap_mean_ci(sub_vals, n_bootstrap, ci_level, seed)
                if ci is not None:
                    ci_key = key.replace("_values", "").replace("aucs", "auc") + "_ci"
                    subset_ci[ci_key] = ci
        if subset_ci:
            res.setdefault("behavior_subsets", {})[subset_name] = subset_ci


def _add_geometric_metrics(
    res: dict,
    explainer,
    id_activations: torch.Tensor,
    ood_activations: torch.Tensor,
    model_type: str,
    bootstrap_repeats: int = 0,
    bootstrap_ci: float = 95.0,
    seed: int = 2026,
) -> None:
    """Compute RAER + hidden-space alignment + energy-weighted alignment and attach to results dict."""
    try:
        res["raer"] = compute_raer(explainer, id_activations, ood_activations, model_type)
        res["hidden_space_alignment"] = compute_hidden_space_alignment(
            explainer, ood_activations, model_type
        )
        res["energy_weighted_alignment"] = compute_energy_weighted_alignment(
            id_activations, ood_activations
        )
    except Exception as e:
        print(f"Warning: failed to compute geometric metrics (RAER / alignment): {e}")

def _append_delta_ce(results, model, explainer, tokens_list, layer, model_type, device=None):
    """Compute Delta CE and merge into results dict."""
    try:
        track_a = os.environ.get("TRACKA_DIAGNOSTIC", "0") == "1"
        dce = compute_delta_ce(
            model=model,
            explainer=explainer,
            tokens_list=tokens_list,
            layer=layer,
            model_type=model_type,
            device=device,
            target_mode="gold",
            batch_size=1,
            return_diagnostics=track_a,
        )
        results["mean_delta_ce"] = dce["mean_delta_ce"]
        results["std_delta_ce"] = dce["std_delta_ce"]
        results["mean_ce_orig"] = dce["mean_ce_orig"]
        results["mean_ce_recon"] = dce["mean_ce_recon"]
        results["delta_ce_n_examples"] = dce["n_examples"]
        if "diagnostics" in dce:
            results["trackA_diagnostics"] = dce["diagnostics"]
            d = dce["diagnostics"]
            print(f"  [TrackA] KL={d['kl_mean']:.4f} top1={d['top1_agree_rate']:.3f} "
                  f"top5={d['top5_agree_rate']:.3f} MSE={d['mse_mean']:.4f} "
                  f"h_norm_ratio={d['h_norm_ratio']:.3f}")
        print(f"  Delta CE = {dce['mean_delta_ce']:.4f} "
              f"(CE_orig={dce['mean_ce_orig']:.4f}, CE_recon={dce['mean_ce_recon']:.4f})")
    except Exception as e:
        print(f"  Warning: Delta CE computation failed: {e}")
    return results


def _append_delta_ce_full_seq(results, model, explainer, tokens_list, layer, model_type, device=None):
    """Compute full-sequence |ΔCE| and merge into results dict with full_seq_* prefixed keys."""
    try:
        track_a = os.environ.get("TRACKA_DIAGNOSTIC", "0") == "1"
        dce = compute_delta_ce_full_seq(
            model=model,
            explainer=explainer,
            tokens_list=tokens_list,
            layer=layer,
            model_type=model_type,
            device=device,
            return_diagnostics=track_a,
        )
        for k, v in dce.items():
            results[k] = v
        print(f"  Full-seq Delta CE = {dce['full_seq_mean_delta_ce']:.4f} "
              f"(CE_orig={dce['full_seq_mean_ce_orig']:.4f}, "
              f"CE_recon={dce['full_seq_mean_ce_recon']:.4f}, "
              f"n_pos={dce['full_seq_n_positions_total']})")
        if "full_seq_diagnostics" in dce:
            d = dce["full_seq_diagnostics"]
            print(f"  [FullSeq TrackA] KL={d['kl_mean']:.4f} top1={d['top1_agree_rate']:.3f} "
                  f"top5={d['top5_agree_rate']:.3f} MSE={d['mse_mean']:.4f} "
                  f"h_norm_ratio={d['h_norm_ratio']:.3f}")
    except Exception as e:
        print(f"  Warning: Full-seq Delta CE computation failed: {e}")
    return results


def _compute_delta_ce_for_all_baselines(
    results, explainer_map, model, eval_tokens_list, layer, model_type, device=None,
):
    """Compute Delta CE (pos=-1 legacy + full-seq primary) for all baselines.

    Note: evaluate_baseline_gae modifies the explainer in-place (adds use_gae,
    B_gae, gae_apply_to etc.).  So explainer_map['fixed'] and ['gae'] may
    point to the same object.  We temporarily disable GAE attributes when
    computing delta CE for the 'fixed' baseline to get a clean measurement.

    Full-seq computation is gated by env var DELTA_CE_FULL_SEQ (default "1"=ON).

    Args:
        results: dict of baseline_name -> metrics_dict
        explainer_map: dict of baseline_name -> explainer object
        model, eval_tokens_list, layer, model_type, device: evaluation params
    """
    if not explainer_map:
        return
    run_full_seq = os.environ.get("DELTA_CE_FULL_SEQ", "1") == "1"
    print("\n" + "=" * 80)
    if run_full_seq:
        print("Computing Delta CE (pos=-1 + full-seq) for all baselines")
    else:
        print("Computing Delta CE (pos=-1 only; DELTA_CE_FULL_SEQ=0) for all baselines")
    print("=" * 80)
    for bl_name, expl in explainer_map.items():
        if bl_name not in results:
            continue
        print(f"\n  [{bl_name}]")
        if bl_name == 'fixed' and getattr(expl, 'use_gae', False):
            # Temporarily disable ALL GAE attributes for clean fixed baseline.
            # _resolve_decoder_override checks gae_decoder_override BEFORE use_gae,
            # so we must clear it too — otherwise GAE decoder leaks into Fixed eval.
            _saved = {}
            for attr in ('use_gae', 'gae_apply_to', 'gae_decoder_override',
                         'gae_decoder_bias', 'gae_feature_adapter', 'gae_feature_gains'):
                _saved[attr] = getattr(expl, attr, None)
                if hasattr(expl, attr):
                    delattr(expl, attr)
            expl.use_gae = False
            try:
                _append_delta_ce(
                    results[bl_name], model, expl, eval_tokens_list, layer, model_type, device,
                )
                if run_full_seq:
                    _append_delta_ce_full_seq(
                        results[bl_name], model, expl, eval_tokens_list, layer, model_type, device,
                    )
            finally:
                for attr, val in _saved.items():
                    if val is not None:
                        setattr(expl, attr, val)
                    elif hasattr(expl, attr):
                        delattr(expl, attr)
        else:
            _append_delta_ce(
                results[bl_name], model, expl, eval_tokens_list, layer, model_type, device,
            )
            if run_full_seq:
                _append_delta_ce_full_seq(
                    results[bl_name], model, expl, eval_tokens_list, layer, model_type, device,
                )


def evaluate_baseline_fixed(
    explainer,
    model,
    tokens_list,
    layer,
    model_type,
    device=None,
    target_mode='argmax',
    batch_id=None,
    target_token_ids=None,
    faith_m_values=None,
    faith_random_repeats=5,
    faith_m_star=32,
    faith_empty_mode="zero_resid",
    faith_rank_mode="causal_topz",
    faith_rank_top_f=64,
    faith_hook_site_mode="auto",
):
    """
    Evaluate Fixed baseline: use ID-trained explainer directly on OOD.
    """
    if device is None:
        device = Config.device
    
    results = evaluate_causal_faithfulness(
        model,
        explainer,
        tokens_list,
        layer,
        model_type,
        device,
        target_mode=target_mode,
        batch_id=batch_id,
        target_token_ids=target_token_ids,
        faith_m_values=faith_m_values,
        faith_random_repeats=faith_random_repeats,
        faith_m_star=faith_m_star,
        faith_empty_mode=faith_empty_mode,
        faith_rank_mode=faith_rank_mode,
        faith_rank_top_f=faith_rank_top_f,
        faith_hook_site_mode=faith_hook_site_mode,
    )
    
    return results


def _iter_streaming_token_batches(
    dataset,
    text_field,
    model,
    context_size,
    batch_size,
    seed=None,
):
    """
    Stream token batches from a streaming HF dataset, similar to SAELens ActivationStore.
    Yields tensors of shape [batch_size, context_size] on CPU.
    """
    if context_size <= 0 or batch_size <= 0:
        raise ValueError("context_size and batch_size must be positive.")

    iterable = dataset
    if hasattr(dataset, "shuffle"):
        try:
            iterable = dataset.shuffle(seed=seed) if seed is not None else dataset
        except Exception:
            iterable = dataset

    data_iter = iter(iterable)
    bos_id = None
    try:
        bos_id = model.tokenizer.bos_token_id
    except Exception:
        bos_id = None

    bos_tensor = None
    if bos_id is not None:
        bos_tensor = torch.tensor([bos_id], dtype=torch.long)

    while True:
        batch_tokens = torch.zeros((0, context_size), dtype=torch.long)
        current_batch = []
        current_length = 0

        while batch_tokens.shape[0] < batch_size:
            try:
                example = next(data_iter)
            except StopIteration:
                data_iter = iter(iterable)
                continue

            source = _extract_text(example, text_field)
            if isinstance(source, str):
                if len(source) == 0:
                    continue
                tokens = model.to_tokens(source, truncate=True, move_to_device=False).squeeze(0)
            elif torch.is_tensor(source):
                tokens = source.to(dtype=torch.long).flatten().cpu()
            elif isinstance(source, (list, tuple)):
                if len(source) == 0:
                    continue
                try:
                    tokens = torch.tensor(source, dtype=torch.long).flatten()
                except Exception:
                    continue
            else:
                continue

            if tokens.numel() == 0:
                continue

            token_len = tokens.shape[0]
            while token_len > 0 and batch_tokens.shape[0] < batch_size:
                space_left = context_size - current_length
                if token_len <= space_left:
                    current_batch.append(tokens[:token_len])
                    current_length += token_len
                    break
                else:
                    current_batch.append(tokens[:space_left])
                    tokens = tokens[space_left:]
                    if bos_tensor is not None:
                        tokens = torch.cat((bos_tensor, tokens), dim=0)
                    token_len = tokens.shape[0]
                    current_length = context_size

                if current_length == context_size:
                    full_batch = torch.cat(current_batch, dim=0)
                    batch_tokens = torch.cat((batch_tokens, full_batch.unsqueeze(0)), dim=0)
                    current_batch = []
                    current_length = 0

        yield batch_tokens[:batch_size]


def _compute_streaming_budget(max_epochs, max_steps, total_tokens, batch_size):
    if total_tokens is not None:
        if total_tokens <= 0:
            raise ValueError("total_tokens must be > 0.")
        total_steps = int(total_tokens // batch_size)
        if total_steps <= 0:
            print(
                "Warning: total_tokens is smaller than batch_size; "
                "running a single step with one full batch."
            )
            total_steps = 1
        total_tokens = total_steps * batch_size
        return total_steps, total_tokens
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        total_steps = int(max_steps)
        total_tokens = total_steps * batch_size
        return total_steps, total_tokens
    if max_epochs <= 0:
        raise ValueError("max_epochs must be > 0.")
    print("Warning: streaming OOD retrain treats max_epochs as max_steps (dataset is streaming).")
    total_steps = int(max_epochs)
    total_tokens = total_steps * batch_size
    return total_steps, total_tokens


def _compute_streaming_budget_tokens_per_step(
    max_epochs: int,
    max_steps: Optional[int],
    total_tokens: Optional[int],
    tokens_per_step: int,
):
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive.")
    if total_tokens is not None:
        if total_tokens <= 0:
            raise ValueError("total_tokens must be > 0.")
        total_steps = int(total_tokens // tokens_per_step)
        if total_steps <= 0:
            print(
                "Warning: total_tokens is smaller than tokens_per_step; "
                "running a single step."
            )
            total_steps = 1
        total_tokens = total_steps * tokens_per_step
        return total_steps, total_tokens
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        total_steps = int(max_steps)
        total_tokens = total_steps * tokens_per_step
        return total_steps, total_tokens
    if max_epochs <= 0:
        raise ValueError("max_epochs must be > 0.")
    print("Warning: streaming residual training treats max_epochs as max_steps.")
    total_steps = int(max_epochs)
    total_tokens = total_steps * tokens_per_step
    return total_steps, total_tokens


def _full_hook_name(hook_name: str, layer: int) -> str:
    if hook_name.startswith("blocks."):
        return hook_name
    if "." not in hook_name and not hook_name.startswith("hook_"):
        hook_name = f"hook_{hook_name}"
    return f"blocks.{layer}.{hook_name}"


def _saeboost_residual_expected_name(model_name: str, explainer_type: str, ood_set: str) -> str:
    return f"{model_name}_{explainer_type}_{ood_set}_saeboost_resid.pt"


def _format_checkpoint_tag(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}".replace("+", "")
    return str(value)


def _ood_retrained_expected_name(
    model_name: str,
    explainer_type: str,
    ood_set: str,
    lr: float,
    lambda_sparse: float,
    total_tokens: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_epochs: Optional[int] = None,
    objective_type: str = "ERM",
    term_t: Optional[float] = None,
) -> str:
    parts = [
        model_name,
        explainer_type,
        ood_set,
        "ood_retrained",
        f"lr{_format_checkpoint_tag(lr)}",
        f"lambda{_format_checkpoint_tag(lambda_sparse)}",
    ]
    if total_tokens is not None:
        parts.append(f"total{int(total_tokens)}")
    elif max_steps is not None:
        parts.append(f"steps{int(max_steps)}")
    elif max_epochs is not None:
        parts.append(f"epochs{int(max_epochs)}")
    if str(objective_type).upper() != "ERM":
        parts.append(f"obj{str(objective_type).lower()}")
        if term_t is not None:
            parts.append(f"t{_format_checkpoint_tag(term_t)}")
    return "_".join(parts) + ".pt"


def _ood_retrained_default_checkpoint_path(
    task_name: str,
    model_name: str,
    explainer_type: str,
    ood_set: str,
    checkpoint_dir: str,
    lr: float,
    lambda_sparse: float,
    total_tokens: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_epochs: Optional[int] = None,
    objective_type: str = "ERM",
    term_t: Optional[float] = None,
) -> str:
    expected_name = _ood_retrained_expected_name(
        model_name=model_name,
        explainer_type=explainer_type,
        ood_set=ood_set,
        lr=lr,
        lambda_sparse=lambda_sparse,
        total_tokens=total_tokens,
        max_steps=max_steps,
        max_epochs=max_epochs,
        objective_type=objective_type,
        term_t=term_t,
    )
    root = Path(checkpoint_dir if checkpoint_dir else "checkpoints")
    if root.name in {task_name, "timeshift_ood", "domain_ood", "adversarial_ood"}:
        return str(root / expected_name)
    return str(root / task_name / expected_name)


def _saeboost_default_residual_checkpoint_path(
    task_name: str,
    model_name: str,
    explainer_type: str,
    ood_set: str,
    residual_checkpoint_dir: str,
) -> str:
    expected_name = _saeboost_residual_expected_name(model_name, explainer_type, ood_set)
    root = Path(residual_checkpoint_dir if residual_checkpoint_dir else "checkpoints")
    return str(root / task_name / expected_name)


def _resolve_saeboost_residual_d_sae(
    base_cfg,
    requested_dict_size: Optional[int] = None,
    default_ratio: float = 0.125,
    min_dict_size: int = 1024,
) -> int:
    d_in = int(base_cfg.d_in)
    base_d_sae = int(getattr(base_cfg, "d_sae", d_in * int(getattr(base_cfg, "expansion_factor", 1))))

    if requested_dict_size is not None:
        if requested_dict_size <= 0:
            raise ValueError("saeboost_residual_dict_size must be positive.")
        target = int(requested_dict_size)
    else:
        target = max(int(round(base_d_sae * default_ratio)), int(min_dict_size))

    expansion_factor = max(1, math.ceil(target / d_in))
    d_sae = expansion_factor * d_in
    if d_sae != target:
        print(
            f"[SAEBoost] Adjusted residual dict_size from requested {target} "
            f"to {d_sae} (must be multiple of d_in={d_in})."
        )
    return d_sae


def _save_saelens_checkpoint(saelens_model, checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": saelens_model.cfg,
            "state_dict": saelens_model.state_dict(),
        },
        str(checkpoint_path),
    )
    print(f"[Checkpoint] Saved SAELens checkpoint: {checkpoint_path}")

    folded_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_folded.pt")
    if hasattr(saelens_model, "set_decoder_norm_to_unit_norm"):
        try:
            original_state = copy.deepcopy(saelens_model.state_dict())
            with torch.no_grad():
                saelens_model.set_decoder_norm_to_unit_norm()
            torch.save(
                {"cfg": saelens_model.cfg, "state_dict": saelens_model.state_dict()},
                str(folded_path),
            )
            saelens_model.load_state_dict(original_state)
            print(f"[Checkpoint] Saved folded SAELens checkpoint: {folded_path}")
        except Exception as e:
            print(f"[Checkpoint] Warning: failed to save folded checkpoint ({e}).")


def _saeboost_batchtopk_feature_acts(feature_acts: torch.Tensor, top_k_per_sample: int) -> torch.Tensor:
    """
    BatchTopK: keep top-(k * batch_size) activations over the flattened batch-feature axis.
    """
    if top_k_per_sample <= 0:
        raise ValueError("top_k_per_sample must be positive for BatchTopK.")
    batch_size = feature_acts.shape[0]
    total_keep = int(top_k_per_sample) * int(batch_size)
    total_keep = min(total_keep, feature_acts.numel())
    if total_keep <= 0:
        return torch.zeros_like(feature_acts)

    flat = feature_acts.flatten()
    topk = torch.topk(flat, total_keep, dim=-1)
    sparse_flat = torch.zeros_like(flat).scatter(-1, topk.indices, topk.values)
    return sparse_flat.reshape_as(feature_acts)


def _forward_residual_saelens(
    saelens_model,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    objective_type: str,
    term_t: float,
    sparse_method: str,
    batch_top_k: Optional[int] = None,
    dead_feature_counters: Optional[torch.Tensor] = None,
    dead_feature_window: int = 10,
    top_k_aux: int = 512,
    aux_penalty: float = 1.0 / 32.0,
):
    """
    Forward pass for residual SAE training with selectable sparsity:
      - l1: default SAELens ReLU + L1 loss
      - batchtopk: SAEBoost BatchTopK-style feature selection
    """
    sparse_method = str(sparse_method).lower()
    base_model = getattr(saelens_model, "_orig_mod", saelens_model)
    if sparse_method == "l1":
        return saelens_model(input_batch, mse_target=target_batch)
    if sparse_method != "batchtopk":
        raise ValueError(f"Unsupported SAEBoost residual sparse method: {sparse_method}")

    x = input_batch.to(base_model.dtype)
    target = target_batch.to(base_model.dtype)
    sae_in = x - base_model.b_dec
    hidden_pre = sae_in @ base_model.W_enc + base_model.b_enc
    dense_feature_acts = torch.relu(hidden_pre)
    feature_acts = _saeboost_batchtopk_feature_acts(dense_feature_acts, int(batch_top_k))

    if getattr(base_model.cfg, "is_transcoder", False):
        sae_out = feature_acts @ base_model.W_dec + base_model.b_dec_out
    else:
        sae_out = feature_acts @ base_model.W_dec + base_model.b_dec

    # SAEBoost-main BatchTopK uses plain L2 (no normalization by target norm).
    residual = target.float() - sae_out.float()
    per_sample_mse = ((sae_out.float() - target.float()) ** 2).mean(dim=-1)
    if objective_type == "ERM":
        mse_loss = erm_loss(per_sample_mse)
    elif objective_type == "TERM":
        mse_loss = term_loss(per_sample_mse, term_t)
    else:
        raise ValueError(f"Unknown objective_type: {objective_type}")

    aux_loss = torch.tensor(0.0, dtype=sae_out.dtype, device=sae_out.device)
    if dead_feature_counters is not None:
        with torch.no_grad():
            fired = dense_feature_acts.sum(dim=0) > 0
            dead_feature_counters[~fired] += 1.0
            dead_feature_counters[fired] = 0.0

        dead_mask = dead_feature_counters >= float(dead_feature_window)
        n_dead = int(dead_mask.sum().item())
        if n_dead > 0:
            k_aux_eff = max(1, min(int(top_k_aux), n_dead))
            dead_acts = dense_feature_acts[:, dead_mask]
            acts_topk_aux = torch.topk(dead_acts, k_aux_eff, dim=-1)
            acts_aux = torch.zeros_like(dead_acts).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ base_model.W_dec[dead_mask, :]
            aux_loss = float(aux_penalty) * ((x_reconstruct_aux.float() - residual) ** 2).mean()

    loss = mse_loss + aux_loss
    l1_loss = torch.tensor(0.0, dtype=sae_out.dtype, device=sae_out.device)
    return sae_out, feature_acts, loss, mse_loss, l1_loss, aux_loss


def _train_saeboost_residual_saelens_streaming(
    base_explainer,
    model: HookedTransformer,
    dataset,
    text_field,
    layer: int,
    context_size: int,
    batch_size: int,
    max_epochs: int,
    max_steps: Optional[int],
    total_tokens: Optional[int],
    lr: float,
    lambda_sparse: float,
    objective_type: str,
    term_t: Optional[float],
    residual_dict_size: Optional[int],
    reinit_b_dec: bool,
    sparse_method: str = "l1",
    residual_batch_top_k: int = 5,
    residual_batch_top_k_start: Optional[int] = None,
    residual_batch_top_k_warmup_fraction: float = 0.1,
    residual_batch_top_k_aux: int = 512,
    residual_batch_top_k_aux_penalty: float = 1.0 / 32.0,
    residual_batch_top_k_dead_feature_window: int = 10,
    dataset_path: str = "custom",
    dataset_name: Optional[str] = None,
    freeze_b_dec: bool = True,
    optimizer_name: str = "adamw",
    scheduler_name: str = "linear",
    scheduler_warmup_fraction: float = 0.1,
    use_torch_compile: bool = True,
    device=None,
    use_wandb: bool = False,
    wandb_prefix: str = "saeboost_residual",
    wandb_log_frequency: int = 50,
):
    if device is None:
        device = Config.device

    if not hasattr(base_explainer, "saelens"):
        raise ValueError("SAEBoost residual training requires a SAELens base explainer.")

    base_saelens = base_explainer.saelens
    base_cfg = base_saelens.cfg
    is_transcoder = bool(getattr(base_cfg, "is_transcoder", False))
    if is_transcoder and getattr(base_explainer, "output_kind", None) != "mlp_out":
        raise NotImplementedError("SAEBoost residual transcoder training supports only output_kind='mlp_out'.")

    if objective_type not in ["ERM", "TERM"]:
        raise ValueError(f"Unsupported objective_type: {objective_type}")
    sparse_method = str(sparse_method).lower()
    if sparse_method not in ["l1", "batchtopk"]:
        raise ValueError(f"Unsupported sparse_method: {sparse_method}")
    if residual_batch_top_k <= 0:
        raise ValueError("residual_batch_top_k must be positive.")
    if residual_batch_top_k_aux <= 0:
        raise ValueError("residual_batch_top_k_aux must be positive.")
    if residual_batch_top_k_dead_feature_window <= 0:
        raise ValueError("residual_batch_top_k_dead_feature_window must be positive.")
    if scheduler_warmup_fraction < 0 or scheduler_warmup_fraction > 1:
        raise ValueError("scheduler_warmup_fraction must be in [0, 1].")
    if optimizer_name.lower() != "adamw":
        raise ValueError("Only optimizer_name='adamw' is supported for SAEBoost residual training.")
    if term_t is None:
        term_t = 1e-3 if is_transcoder else 1.0

    # One optimization step consumes `train_batch_size` activation vectors from SAELens ActivationsStore.
    # (Not `batch_size * context_size` raw text tokens.)
    tokens_per_step = int(batch_size)
    total_steps, total_tokens = _compute_streaming_budget_tokens_per_step(
        max_epochs=max_epochs,
        max_steps=max_steps,
        total_tokens=total_tokens,
        tokens_per_step=tokens_per_step,
    )
    print(
        "[SAEBoost] residual training budget: "
        f"total_steps={total_steps}, total_tokens={total_tokens}, "
        f"batch_size={batch_size}, context_size={context_size}, "
        f"tokens_per_step={tokens_per_step} (activation vectors)"
    )

    default_input_hook = "ln2.hook_normalized" if is_transcoder else "resid_post"
    input_hook = getattr(base_explainer, "input_hook_point", None) or default_input_hook
    output_hook = getattr(base_explainer, "output_hook_point", None) or "hook_mlp_out"
    hook_point_full = _full_hook_name(input_hook, layer)
    output_hook_full = _full_hook_name(output_hook, layer)

    # Build residual SAE/Transcoder config from base config with a smaller dictionary.
    cfg = copy.deepcopy(base_cfg)
    cfg.device = str(device)
    cfg.objective_type = objective_type
    cfg.term_t = term_t
    cfg.l1_coefficient = float(lambda_sparse)
    cfg.lr = float(lr)
    cfg.context_size = int(context_size)
    cfg.train_batch_size = int(batch_size)
    cfg.total_training_tokens = int(total_tokens)
    cfg.reinit_b_dec = bool(reinit_b_dec)
    cfg.hook_point = hook_point_full
    cfg.hook_point_layer = int(layer)
    cfg.dataset_path = dataset_path
    cfg.dataset_name = dataset_name
    cfg.dataset_override = dataset
    cfg.dataset_text_field = text_field
    cfg.is_dataset_tokenized = False
    cfg.use_cached_activations = False
    cfg.is_transcoder = bool(is_transcoder)
    if is_transcoder:
        cfg.out_hook_point = output_hook_full
        cfg.out_hook_point_layer = int(layer)
        cfg.d_out = int(cfg.d_in)
    else:
        cfg.out_hook_point = None
        cfg.out_hook_point_layer = None
        cfg.d_out = None

    store_batch_size = Config.SAELENS_STORE_BATCH_SIZE
    if store_batch_size is None:
        store_batch_size = min(32, max(1, int(batch_size)))
    n_batches_in_buffer = Config.SAELENS_N_BATCHES_IN_BUFFER
    if n_batches_in_buffer is None:
        n_batches_in_buffer = 64
    cfg.store_batch_size = int(max(1, store_batch_size))
    cfg.n_batches_in_buffer = int(max(2, n_batches_in_buffer))
    cfg.use_ghost_grads = False
    cfg.feature_sampling_method = None
    cfg.saeboost_sparse_method = str(sparse_method).lower()
    cfg.saeboost_inference_sparse_method = (
        "batchtopk" if str(sparse_method).lower() == "batchtopk" else "none"
    )

    resid_d_sae = _resolve_saeboost_residual_d_sae(
        base_cfg=base_cfg,
        requested_dict_size=residual_dict_size,
    )
    cfg.expansion_factor = max(1, resid_d_sae // int(cfg.d_in))
    cfg.d_sae = int(cfg.d_in) * int(cfg.expansion_factor)

    residual_saelens = SaeLensSparseAutoencoder(cfg)
    residual_saelens.to(device)
    residual_saelens.train()
    if freeze_b_dec:
        with torch.no_grad():
            if hasattr(residual_saelens, "b_dec") and residual_saelens.b_dec is not None:
                base_b_dec = getattr(base_saelens, "b_dec", None)
                if base_b_dec is not None and tuple(base_b_dec.shape) == tuple(residual_saelens.b_dec.shape):
                    residual_saelens.b_dec.data.copy_(base_b_dec.detach().to(residual_saelens.b_dec.device))
                else:
                    residual_saelens.b_dec.data.zero_()
                residual_saelens.b_dec.requires_grad_(False)
            if hasattr(residual_saelens, "b_dec_out") and residual_saelens.b_dec_out is not None:
                base_b_dec_out = getattr(base_saelens, "b_dec_out", None)
                if base_b_dec_out is not None and tuple(base_b_dec_out.shape) == tuple(residual_saelens.b_dec_out.shape):
                    residual_saelens.b_dec_out.data.copy_(base_b_dec_out.detach().to(residual_saelens.b_dec_out.device))
                else:
                    residual_saelens.b_dec_out.data.zero_()
                residual_saelens.b_dec_out.requires_grad_(False)
        print("[SAEBoost] Frozen residual decoder bias terms with base-stat initialization when available.")

    base_explainer.to(device)
    base_explainer.eval()
    for p in base_explainer.parameters():
        p.requires_grad_(False)

    model.to(device)
    model.eval()

    trainable_params = [p for p in residual_saelens.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )
    total_steps_for_sched = max(1, total_steps)
    warmup_steps = min(int(total_steps_for_sched * float(scheduler_warmup_fraction)), total_steps_for_sched)
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps_for_sched,
    )
    base_forward_model = base_explainer
    residual_forward_model = residual_saelens
    if use_torch_compile and hasattr(torch, "compile"):
        try:
            base_forward_model = torch.compile(base_explainer)
            print("[SAEBoost] torch.compile enabled for base explainer.")
        except Exception as e:
            print(f"[SAEBoost] Warning: torch.compile(base_explainer) failed ({e}).")
        try:
            residual_forward_model = torch.compile(residual_saelens)
            print("[SAEBoost] torch.compile enabled for residual model.")
        except Exception as e:
            print(f"[SAEBoost] Warning: torch.compile(residual model) failed ({e}).")

    batch_top_k_start = residual_batch_top_k_start
    if batch_top_k_start is None:
        # Dense start by default: start_top_k equals the actual residual dictionary size.
        batch_top_k_start = int(residual_saelens.d_sae)
    else:
        batch_top_k_start = int(batch_top_k_start)
        # If user passed requested dict_size (e.g., 1024) but actual d_sae was rounded up
        # (e.g., 1536), align start_top_k with actual d_sae for SAEBoost-style dense warmup.
        if (
            residual_dict_size is not None
            and batch_top_k_start == int(residual_dict_size)
            and int(residual_saelens.d_sae) != batch_top_k_start
        ):
            print(
                "[SAEBoost] Adjusted batch_top_k_start from requested "
                f"{batch_top_k_start} to actual residual d_sae={int(residual_saelens.d_sae)} "
                "for dense warmup."
            )
            batch_top_k_start = int(residual_saelens.d_sae)
    batch_top_k_start = max(1, min(batch_top_k_start, int(residual_saelens.d_sae)))
    batch_top_k_warmup_steps = max(
        0, int(total_steps * float(residual_batch_top_k_warmup_fraction))
    )
    if sparse_method == "batchtopk":
        cfg.saeboost_batch_top_k = int(residual_batch_top_k)
        cfg.saeboost_batch_top_k_start = int(batch_top_k_start)
        cfg.saeboost_batch_top_k_warmup_fraction = float(residual_batch_top_k_warmup_fraction)
    else:
        cfg.saeboost_batch_top_k = None
        cfg.saeboost_batch_top_k_start = None
        cfg.saeboost_batch_top_k_warmup_fraction = None
    cfg.saeboost_jumprelu_threshold = None
    if sparse_method == "batchtopk":
        print(
            "[SAEBoost] BatchTopK aux settings: "
            f"top_k_aux={residual_batch_top_k_aux}, aux_penalty={residual_batch_top_k_aux_penalty}, "
            f"dead_feature_window={residual_batch_top_k_dead_feature_window}"
        )
        print(
            "[SAEBoost] BatchTopK residual training: "
            f"top_k={residual_batch_top_k}, start_top_k={batch_top_k_start}, "
            f"warmup_steps={batch_top_k_warmup_steps}"
        )
    dead_feature_counters = None
    if sparse_method == "batchtopk":
        dead_feature_counters = torch.zeros(
            residual_saelens.d_sae, device=device, dtype=torch.float32
        )

    activation_store = ActivationsStore(cfg, model)
    print(
        "[SAEBoost] ActivationStore enabled: "
        f"hook={hook_point_full}, "
        f"out_hook={output_hook_full if is_transcoder else 'N/A'}, "
        f"store_batch_size={cfg.store_batch_size}, n_batches_in_buffer={cfg.n_batches_in_buffer}"
    )

    if Config.device.type == "cuda":
        torch.cuda.synchronize()
    pbar = tqdm(total=total_tokens, desc="Training SAEBoost residual")
    for step in range(total_steps):
        batch_data = activation_store.next_batch()
        if batch_data.device != device:
            batch_data = batch_data.to(device, non_blocking=True)
        batch_data = batch_data.to(dtype=residual_saelens.dtype)

        with torch.no_grad():
            if is_transcoder:
                input_batch = batch_data[:, : cfg.d_in]
                output_batch = batch_data[:, cfg.d_in : cfg.d_in + cfg.d_out]
                y_base = base_forward_model(input_batch)
                target_batch = output_batch - y_base
            else:
                input_batch = batch_data
                h_base = base_forward_model(input_batch)
                target_batch = input_batch - h_base

        current_top_k = residual_batch_top_k
        if sparse_method == "batchtopk" and batch_top_k_warmup_steps > 0 and step < batch_top_k_warmup_steps:
            ratio = float(step) / float(max(1, batch_top_k_warmup_steps))
            current_top_k = int(
                round(batch_top_k_start - (batch_top_k_start - residual_batch_top_k) * ratio)
            )
            current_top_k = max(1, current_top_k)

        optimizer.zero_grad()
        _, _, loss, mse_loss, l1_loss, extra_loss = _forward_residual_saelens(
            residual_forward_model,
            input_batch=input_batch,
            target_batch=target_batch,
            objective_type=objective_type,
            term_t=term_t,
            sparse_method=sparse_method,
            batch_top_k=current_top_k,
            dead_feature_counters=dead_feature_counters,
            dead_feature_window=residual_batch_top_k_dead_feature_window,
            top_k_aux=residual_batch_top_k_aux,
            aux_penalty=residual_batch_top_k_aux_penalty,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_saelens.parameters(), max_norm=1.0)
        if hasattr(residual_saelens, "make_decoder_weights_and_grad_unit_norm"):
            residual_saelens.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        scheduler.step()

        if (step + 1) % max(1, wandb_log_frequency) == 0 or step == 0:
            if sparse_method == "batchtopk":
                pbar.set_postfix(
                    loss=f"{loss.item():.4e}",
                    mse=f"{mse_loss.item():.4e}",
                    aux=f"{extra_loss.item():.4e}",
                    k=int(current_top_k),
                )
            else:
                pbar.set_postfix(
                    loss=f"{loss.item():.4e}",
                    mse=f"{mse_loss.item():.4e}",
                    l1=f"{l1_loss.item():.4e}",
                )

        pbar.update(tokens_per_step)
        if use_wandb and WANDB_AVAILABLE and (step + 1) % wandb_log_frequency == 0:
            prefix = f"{wandb_prefix}/" if wandb_prefix else ""
            log_payload = {
                f"{prefix}train/total_loss": loss.item(),
                f"{prefix}train/mse_loss": mse_loss.item(),
                f"{prefix}train/l1_loss": l1_loss.item(),
                f"{prefix}train/learning_rate": optimizer.param_groups[0]["lr"],
                f"{prefix}step": step + 1,
            }
            if sparse_method == "batchtopk":
                log_payload[f"{prefix}train/aux_loss"] = extra_loss.item()
                log_payload[f"{prefix}train/current_top_k"] = int(current_top_k)
            wandb.log(log_payload)
    pbar.close()
    residual_saelens.eval()
    return residual_saelens




def _train_transcoder_saelens_streaming(
    model: HookedTransformer,
    model_name: str,
    dataset_path: str,
    dataset_name: str,
    dataset_override,
    dataset_text_field,
    layer: int,
    context_size: int,
    batch_size: int,
    total_tokens: int,
    lr: float,
    lambda_sparse: float,
    objective_type: str,
    term_t: float,
    device=None,
    use_wandb: bool = False,
    output_kind: str = None,
    input_hook_point: str = None,
    output_hook_point: str = None,
    warm_start_path: str = None,
    warm_start_state_dict: dict = None,
    reinit_b_dec: bool = True,
    lr_scheduler_name: Optional[str] = None,
    lr_warm_up_steps: Optional[int] = None,
):
    if output_kind != "mlp_out":
        raise NotImplementedError("SAELens OOD retrain is only supported for mlp_out transcoders.")
    if device is None:
        device = Config.device

    model.to(device)
    model.eval()

    d_in = model.cfg.d_model
    dict_size_k = Config.get_dict_size_k(model_name)
    if dict_size_k % d_in != 0:
        raise ValueError("dict_size_k must be a multiple of d_model to match SAELens expansion_factor.")
    expansion_factor = dict_size_k // d_in

    if input_hook_point is None:
        input_hook_point = "ln2.hook_normalized"
    if output_hook_point is None:
        output_hook_point = "hook_mlp_out"

    def _full_hook(hook_name: str, layer_idx: int) -> str:
        if hook_name.startswith("blocks."):
            return hook_name
        if "." not in hook_name and not hook_name.startswith("hook_"):
            hook_name = f"hook_{hook_name}"
        return f"blocks.{layer_idx}.{hook_name}"

    hook_point = _full_hook(input_hook_point, layer)
    out_hook_point = _full_hook(output_hook_point, layer)

    if Config.USE_BF16:
        cfg_dtype = torch.bfloat16
    elif Config.USE_FP16:
        cfg_dtype = torch.float16
    else:
        cfg_dtype = torch.float32

    dict_size_k = expansion_factor * d_in
    large_dict = dict_size_k >= 65536 or d_in >= 2048
    store_batch_size = Config.SAELENS_STORE_BATCH_SIZE
    if store_batch_size is None:
        store_batch_size = min(4, batch_size) if large_dict else min(32, batch_size)
    n_batches_in_buffer = Config.SAELENS_N_BATCHES_IN_BUFFER
    if n_batches_in_buffer is None:
        n_batches_in_buffer = 8 if large_dict else 64
    store_batch_size = max(1, int(store_batch_size))
    n_batches_in_buffer = max(1, int(n_batches_in_buffer))

    cfg = LanguageModelSAERunnerConfig(
        model_name=model_name,
        hook_point=hook_point,
        hook_point_layer=layer,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_override=dataset_override,
        dataset_text_field=dataset_text_field,
        is_dataset_tokenized=False,
        context_size=context_size,
        d_in=d_in,
        expansion_factor=expansion_factor,
        b_dec_init_method="mean",
        reinit_b_dec=reinit_b_dec,
        lr=lr,
        l1_coefficient=lambda_sparse,
        objective_type=objective_type,
        term_t=term_t,
        lr_scheduler_name=(lr_scheduler_name if lr_scheduler_name is not None else "constantwithwarmup"),
        lr_warm_up_steps=(lr_warm_up_steps if lr_warm_up_steps is not None else 5000),
        train_batch_size=batch_size,
        n_batches_in_buffer=n_batches_in_buffer,
        total_training_tokens=total_tokens,
        store_batch_size=store_batch_size,
        use_ghost_grads=False,
        feature_sampling_method=None,
        feature_sampling_window=1000,
        resample_batches=32,
        dead_feature_window=5000,
        dead_feature_threshold=1e-8,
        log_to_wandb=use_wandb,
        wandb_project="GAE_OOD-Retrain",
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        device=str(device),
        seed=Config.SEED,
        dtype=cfg_dtype,
        is_transcoder=True,
        out_hook_point=out_hook_point,
        out_hook_point_layer=layer,
        d_out=d_in,
    )

    print(
        "[SAELens] buffer settings "
        f"store_batch_size={store_batch_size}, n_batches_in_buffer={n_batches_in_buffer}, dtype={cfg_dtype}"
    )

    activation_store = ActivationsStore(cfg, model)
    sparse_autoencoder = SaeLensSparseAutoencoder(cfg)

    if warm_start_state_dict is not None:
        try:
            sparse_autoencoder.load_state_dict(warm_start_state_dict, strict=False)
            print("Warm-started SAELens transcoder from in-memory state_dict")
        except Exception as e:
            print(f"Warning: warm-start (state_dict) failed ({e}). Training from scratch.")
    elif warm_start_path:
        try:
            state = torch.load(warm_start_path, map_location=str(device), weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                sparse_autoencoder.load_state_dict(state["state_dict"], strict=False)
                print(f"Warm-started SAELens transcoder from {warm_start_path}")
        except Exception as e:
            print(f"Warning: warm-start failed ({e}). Training from scratch.")

    sparse_autoencoder = train_sae_on_language_model(
        model,
        sparse_autoencoder,
        activation_store,
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size,
        feature_sampling_method=cfg.feature_sampling_method,
        feature_sampling_window=cfg.feature_sampling_window,
        feature_reinit_scale=cfg.feature_reinit_scale,
        dead_feature_threshold=cfg.dead_feature_threshold,
        dead_feature_window=cfg.dead_feature_window,
        use_wandb=cfg.log_to_wandb and WANDB_AVAILABLE,
        wandb_log_frequency=cfg.wandb_log_frequency,
    )

    return sparse_autoencoder




def _train_saelens_streaming(
    saelens_model,
    model,
    dataset,
    text_field,
    layer,
    context_size=128,
    batch_size=64,
    max_epochs=50,
    max_steps=None,
    total_tokens=None,
    lr=2e-5,
    objective_type="ERM",
    term_t=1e-3,
    lambda_sparse=1e-4,
    device=None,
    use_wandb=False,
    wandb_prefix="",
    wandb_log_frequency=50,
    output_kind=None,
    input_hook_point=None,
    output_hook_point=None,
):
    if device is None:
        device = Config.device

    # Update cfg to reflect retraining objective/coefficients
    try:
        saelens_model.cfg.objective_type = objective_type
        saelens_model.cfg.term_t = term_t
        saelens_model.cfg.l1_coefficient = lambda_sparse
    except Exception:
        pass

    saelens_model.to(device)
    saelens_model.train()

    model.to(device)
    model.eval()

    optimizer = torch.optim.Adam(saelens_model.parameters(), lr=lr)
    # For SAE all-positions extraction: 1 LM forward yields batch*ctx SAE samples.
    # For mlp_out (transcoder, position=-1): 1 LM forward yields batch SAE samples.
    is_sae_all_positions = (output_kind != "mlp_out")
    samples_per_step = (batch_size * context_size) if is_sae_all_positions else batch_size
    total_steps, total_tokens = _compute_streaming_budget(
        max_epochs, max_steps, total_tokens, samples_per_step
    )
    print(
        f"Training budget: total_steps={total_steps}, total_tokens={total_tokens}, "
        f"batch_size={batch_size}, context_size={context_size}, "
        f"samples_per_step={samples_per_step}"
    )

    warmup_steps = 5000
    total_steps_for_sched = max(1, total_steps)
    warmup_steps = min(warmup_steps, total_steps_for_sched)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if total_steps_for_sched <= warmup_steps:
            return 2e-6 / lr
        progress = (step - warmup_steps) / (total_steps_for_sched - warmup_steps)
        return 2e-6 / lr + (1 - 2e-6 / lr) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    data_iter = _iter_streaming_token_batches(
        dataset, text_field, model, context_size, batch_size, seed=Config.SEED
    )

    if Config.device.type == "cuda":
        torch.cuda.synchronize()
    pbar = tqdm(total=total_tokens, desc="Training SAELens (streaming)")
    for step in range(total_steps):
        tokens = next(data_iter).to(device, non_blocking=True)

        with torch.no_grad():
            if output_kind == "mlp_out":
                h_in, h_out = extract_mlp_in_out_activations(
                    model,
                    tokens,
                    layer,
                    position=-1,
                    hook_in=input_hook_point or "ln2.hook_normalized",
                    hook_out=output_hook_point or "hook_mlp_out",
                )
                h_batch = h_in
                target_batch = h_out
            else:
                # SAE path: extract from input_hook_point at ALL positions to amortize LM forward.
                # Yields batch*ctx SAE samples per step (vs batch with position=-1).
                _hook_name = (input_hook_point or "ln2.hook_normalized")
                if _hook_name.startswith("blocks."):
                    _hook_full = _hook_name
                elif "." in _hook_name:
                    _hook_full = f"blocks.{layer}.{_hook_name}"
                else:
                    if not _hook_name.startswith("hook_"):
                        _hook_name = f"hook_{_hook_name}"
                    _hook_full = f"blocks.{layer}.{_hook_name}"
                _captured = []
                def _all_pos_hook(value, hook):
                    # value: [batch, seq, d_model] -> reshape to [batch*seq, d_model]
                    _captured.append(value.detach().reshape(-1, value.shape[-1]).clone())
                    return value
                model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(_hook_full, _all_pos_hook)],
                    stop_at_layer=layer + 1,
                )
                h_batch = _captured[0]
                target_batch = None

        optimizer.zero_grad()
        if target_batch is not None:
            _, _, loss, mse_loss, l1_loss, _ = saelens_model(h_batch, mse_target=target_batch)
        else:
            _, _, loss, mse_loss, l1_loss, _ = saelens_model(h_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(saelens_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        pbar.update(samples_per_step)
        if use_wandb and WANDB_AVAILABLE and (step + 1) % wandb_log_frequency == 0:
            prefix = f"{wandb_prefix}/" if wandb_prefix else ""
            wandb.log(
                {
                    f"{prefix}train/total_loss": loss.item(),
                    f"{prefix}train/mse_loss": mse_loss.item(),
                    f"{prefix}train/l1_loss": l1_loss.item(),
                    f"{prefix}train/learning_rate": optimizer.param_groups[0]['lr'],
                    f"{prefix}step": step + 1,
                }
            )
    pbar.close()


def _train_saelens_batch(
    saelens_model,
    activations,
    targets=None,
    max_epochs=100,
    max_steps=None,
    total_tokens=None,
    lr=2e-5,
    objective_type="ERM",
    term_t=1e-3,
    lambda_sparse=1e-4,
    device=None,
    use_wandb=False,
    wandb_prefix="",
    use_cosine_scheduler=True,
    eta_min=2e-6,
    batch_size=None,
    wandb_log_frequency=50,
):
    if device is None:
        device = Config.device

    try:
        saelens_model.cfg.objective_type = objective_type
        saelens_model.cfg.term_t = term_t
        saelens_model.cfg.l1_coefficient = lambda_sparse
    except Exception:
        pass

    saelens_model.to(device)
    saelens_model.train()

    optimizer = torch.optim.Adam(saelens_model.parameters(), lr=lr)

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if targets is None:
        dataset = torch.utils.data.TensorDataset(activations)
    else:
        dataset = torch.utils.data.TensorDataset(activations, targets)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("No training batches available. Check activations and batch_size.")

    if total_tokens is not None:
        if total_tokens <= 0:
            raise ValueError("total_tokens must be > 0.")
        total_steps = int(total_tokens // batch_size)
        if total_steps <= 0:
            total_steps = 1
        total_tokens = total_steps * batch_size
    elif max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0.")
        total_steps = int(max_steps)
        total_tokens = total_steps * batch_size
    else:
        total_steps = int(max_epochs * steps_per_epoch)
        total_tokens = total_steps * batch_size
    print(f"Training budget: total_steps={total_steps}, total_tokens={total_tokens}, batch_size={batch_size}")

    warmup_steps = 5000
    if use_cosine_scheduler:
        total_steps_for_sched = max(1, total_steps)
        warmup_steps = min(warmup_steps, total_steps_for_sched)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            if total_steps_for_sched <= warmup_steps:
                return eta_min / lr
            progress = (step - warmup_steps) / (total_steps_for_sched - warmup_steps)
            return eta_min / lr + (1 - eta_min / lr) * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    data_iter = iter(train_loader)
    if Config.device.type == "cuda":
        torch.cuda.synchronize()
    pbar = tqdm(total=total_tokens, desc="Training SAELens")
    for step in range(total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if targets is None:
            (h_batch,) = batch
            h_batch = h_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            _, _, loss, mse_loss, l1_loss, _ = saelens_model(h_batch)
        else:
            h_batch, target_batch = batch
            h_batch = h_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            _, _, loss, mse_loss, l1_loss, _ = saelens_model(h_batch, mse_target=target_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(saelens_model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        pbar.update(batch_size)
        if use_wandb and WANDB_AVAILABLE and (step + 1) % wandb_log_frequency == 0:
            prefix = f"{wandb_prefix}/" if wandb_prefix else ""
            wandb.log(
                {
                    f"{prefix}train/total_loss": loss.item(),
                    f"{prefix}train/mse_loss": mse_loss.item(),
                    f"{prefix}train/l1_loss": l1_loss.item(),
                    f"{prefix}train/learning_rate": optimizer.param_groups[0]['lr'],
                    f"{prefix}step": step + 1,
                }
            )
    pbar.close()

def evaluate_baseline_ood_retrained(
    model,
    ood_activations,
    ood_logits,
    layer,
    model_type,
    d_model,
    vocab_size=None,
    max_epochs=50,
    max_steps=None,
    total_tokens=None,
    lr=1e-3,
    lambda_sparse=0.01,
    objective_type="ERM",
    term_t=None,
    batch_size=None,
    use_streaming=False,
    ood_dataset=None,
    ood_text_field=None,
    ood_dataset_path=None,
    ood_dataset_name=None,
    context_size=128,
    input_hook_point=None,
    output_hook_point=None,
    warm_start=False,
    reinit_b_dec=True,
    warm_start_explainer=None,
    warm_start_path=None,
    use_saelens=True,
    device=None,
    use_wandb=False,
    output_kind: str = None,
    task_name: str = "timeshift_ood",
    ood_set: Optional[str] = None,
    base_checkpoint_path: Optional[str] = None,
    retrained_checkpoint_path: Optional[str] = None,
    retrained_checkpoint_dir: str = os.environ.get("REPO_DATA", "./data") + "/checkpoints",
    lr_scheduler_name: Optional[str] = None,
    lr_warm_up_steps: Optional[int] = None,
):
    """
    Evaluate OOD-retrained baseline: retrain explainer on OOD activations.

    `lr_scheduler_name` / `lr_warm_up_steps` are optional overrides for the
    underlying SAELens training loop. When None, the loop uses its default
    ('constantwithwarmup' with 5000 warmup steps) which matches the ID
    pretraining recipe. Override them to match a short-finetune recipe
    (e.g. 'linearwarmupdecay' with a small 5%-of-budget warmup) when warm-
    starting from a well-trained checkpoint.
    """
    if device is None:
        device = Config.device

    model_name = None
    for name, path in Config.MODELS.items():
        if path in str(model.cfg.model_name) or name in str(model.cfg.model_name):
            model_name = name
            break
    if model_name is None:
        model_name = "gpt2"
    if not use_saelens:
        raise ValueError("SAELens-only OOD retraining is enabled; disabling SAELens is not supported.")

    if not isinstance(warm_start_explainer, (SaeLensTranscoderAdapter, SaeLensSparseAutoencoderAdapter)):
        raise ValueError("SAELens retrain requires a SAELens explainer checkpoint.")

    is_transcoder = isinstance(warm_start_explainer, SaeLensTranscoderAdapter)

    save_path = None
    if ood_set:
        if retrained_checkpoint_path is not None:
            save_path = retrained_checkpoint_path
        else:
            save_path = _ood_retrained_default_checkpoint_path(
                task_name=task_name,
                model_name=model_name,
                explainer_type=model_type,
                ood_set=ood_set,
                checkpoint_dir=retrained_checkpoint_dir,
                lr=lr,
                lambda_sparse=lambda_sparse,
                total_tokens=total_tokens,
                max_steps=max_steps,
                max_epochs=max_epochs,
                objective_type=objective_type,
                term_t=term_t,
            )
        if Path(save_path).exists():
            print(f"[OOD-Retrained] Loading cached checkpoint: {save_path}")
            return load_explainer(save_path, model_type, d_model, vocab_size)

    # Retrain on OOD
    budget_bits = []
    if total_tokens is not None:
        budget_bits.append(f"total_tokens={total_tokens}")
    if max_steps is not None:
        budget_bits.append(f"max_steps={max_steps}")
    if total_tokens is None and max_steps is None:
        budget_bits.append(f"max_epochs={max_epochs}")
    if batch_size is not None:
        budget_bits.append(f"batch_size={batch_size}")
    budget_str = ", ".join(budget_bits) if budget_bits else f"max_epochs={max_epochs}"
    print(
        "Retraining explainer on OOD activations "
        f"({budget_str}, lr={lr}, lambda_sparse={lambda_sparse}, objective={objective_type})..."
    )
    if term_t is None:
        term_t = 1e-3 if is_transcoder else 1.0

    if use_streaming:
        if ood_dataset is None or ood_text_field is None:
            raise ValueError("Streaming OOD retrain requires ood_dataset and ood_text_field.")
        if batch_size is None:
            batch_size = Config.BATCH_SIZE

        if is_transcoder and output_kind == "mlp_out":
            total_steps, total_tokens = _compute_streaming_budget(
                max_epochs, max_steps, total_tokens, batch_size
            )
            print(
                "Retraining explainer on OOD activations (SAELens) "
                f"(total_steps={total_steps}, total_tokens={total_tokens}, "
                f"batch_size={batch_size}, context_size={context_size}, lr={lr}, lambda_sparse={lambda_sparse}, "
                f"objective={objective_type})..."
            )
            dataset_path = ood_dataset_path or "custom"
            warm_start_state = None
            if warm_start:
                warm_start_state = warm_start_explainer.saelens.state_dict()
                print("Warm-started SAELens transcoder from ID checkpoint (state_dict)")
            saelens = _train_transcoder_saelens_streaming(
                model=model,
                model_name=model_name,
                dataset_path=dataset_path,
                dataset_name=ood_dataset_name,
                dataset_override=ood_dataset,
                dataset_text_field=ood_text_field,
                layer=layer,
                context_size=context_size,
                batch_size=batch_size,
                total_tokens=total_tokens,
                lr=lr,
                lambda_sparse=lambda_sparse,
                objective_type=objective_type,
                term_t=term_t,
                device=device,
                use_wandb=use_wandb,
                output_kind=output_kind,
                input_hook_point=input_hook_point,
                output_hook_point=output_hook_point,
                warm_start_path=warm_start_path if warm_start else None,
                warm_start_state_dict=warm_start_state,
                reinit_b_dec=reinit_b_dec,
                lr_scheduler_name=lr_scheduler_name,
                lr_warm_up_steps=lr_warm_up_steps,
            )
            if save_path:
                _save_saelens_checkpoint(saelens, save_path)
            return SaeLensTranscoderAdapter(saelens)

        # SAE (or non-mlp_out) streaming retrain using SAELens model
        cfg = warm_start_explainer.saelens.cfg
        try:
            import copy
            cfg = copy.deepcopy(cfg)
        except Exception:
            pass
        cfg.device = str(device)
        saelens_model = SaeLensSparseAutoencoder(cfg)
        if warm_start:
            saelens_model.load_state_dict(warm_start_explainer.saelens.state_dict(), strict=False)
            if is_transcoder:
                print("Warm-started SAELens transcoder from ID checkpoint (state_dict)")
        saelens_model.to(device)
        saelens_model.eval()

        _train_saelens_streaming(
            saelens_model,
            model,
            ood_dataset,
            ood_text_field,
            layer,
            context_size=context_size,
            batch_size=batch_size,
            max_epochs=max_epochs,
            max_steps=max_steps,
            total_tokens=total_tokens,
            lr=lr,
            objective_type=objective_type,
            term_t=term_t,
            lambda_sparse=lambda_sparse,
            device=device,
            use_wandb=use_wandb,
            wandb_prefix="ood_retrained",
            output_kind=output_kind,
            input_hook_point=input_hook_point,
            output_hook_point=output_hook_point,
        )

        if save_path:
            _save_saelens_checkpoint(saelens_model, save_path)

        if is_transcoder:
            return SaeLensTranscoderAdapter(saelens_model)
        return SaeLensSparseAutoencoderAdapter(saelens_model)

    # Non-streaming retrain with cached activations
    cfg = warm_start_explainer.saelens.cfg
    try:
        import copy
        cfg = copy.deepcopy(cfg)
    except Exception:
        pass
    cfg.device = str(device)
    saelens_model = SaeLensSparseAutoencoder(cfg)
    if warm_start:
        saelens_model.load_state_dict(warm_start_explainer.saelens.state_dict(), strict=False)
        if is_transcoder:
            print("Warm-started SAELens transcoder from ID checkpoint (state_dict)")
    saelens_model.to(device)
    saelens_model.eval()

    targets = None
    if is_transcoder:
        targets = ood_logits
    _train_saelens_batch(
        saelens_model,
        ood_activations,
        targets=targets,
        max_epochs=max_epochs,
        max_steps=max_steps,
        total_tokens=total_tokens,
        lr=lr,
        objective_type=objective_type,
        term_t=term_t,
        lambda_sparse=lambda_sparse,
        device=device,
        use_wandb=use_wandb,
        wandb_prefix="ood_retrained",
        batch_size=batch_size,
    )

    if save_path:
        _save_saelens_checkpoint(saelens_model, save_path)

    if is_transcoder:
        return SaeLensTranscoderAdapter(saelens_model)
    return SaeLensSparseAutoencoderAdapter(saelens_model)
    
    # The function always returns above.


def evaluate_baseline_ood_finetuned(
    model,
    ood_activations,
    ood_logits,
    layer,
    model_type,
    d_model,
    vocab_size=None,
    *,
    # Kissane recipe defaults — can be overridden per CLI
    total_tokens: int = 5_000_000,
    lr: Optional[float] = None,
    lambda_sparse: Optional[float] = None,
    batch_size: int = 256,
    context_size: int = 128,
    loss_type: str = "mse",
    # Shared plumbing (mirrors ood_retrained signature)
    ood_dataset=None,
    ood_text_field=None,
    ood_dataset_path=None,
    ood_dataset_name=None,
    input_hook_point=None,
    output_hook_point=None,
    warm_start_explainer=None,
    warm_start_path=None,
    device=None,
    use_wandb=False,
    output_kind: str = None,
    task_name: str = "timeshift_ood",
    ood_set: Optional[str] = None,
    base_checkpoint_path: Optional[str] = None,
    finetuned_checkpoint_path: Optional[str] = None,
    finetuned_checkpoint_dir: str = os.environ.get("REPO_DATA", "./data") + "/checkpoints/ood_finetuned",
):
    """
    Thin wrapper over `evaluate_baseline_ood_retrained` that enforces the
    Kissane 2024 fine-tune recipe: warm-start from the ID checkpoint,
    disable b_dec reinit, stream OOD activations, and run a short (~5M
    token) fine-tune with MSE+L1 loss at the ID pretraining LR.

    Only `loss_type="mse"` is implemented (Variant A). `loss_type="kl"`
    (Variant B, Karvonen 2025) is reserved as a placeholder.
    """
    if loss_type != "mse":
        raise NotImplementedError(
            f"ood_finetuned loss_type={loss_type!r} not implemented; "
            "only 'mse' (Variant A, Kissane recipe) is supported."
        )

    # lr / lambda_sparse default to whatever the ID warm-start explainer used.
    if lr is None:
        cfg = getattr(getattr(warm_start_explainer, "saelens", None), "cfg", None)
        lr = float(getattr(cfg, "lr", 1e-4)) if cfg is not None else 1e-4
    if lambda_sparse is None:
        cfg = getattr(getattr(warm_start_explainer, "saelens", None), "cfg", None)
        lambda_sparse = float(getattr(cfg, "l1_coefficient", 1e-5)) if cfg is not None else 1e-5

    # Kissane recipe LR schedule: linear warmup 5% -> linear decay to zero.
    # The underlying training loop defaults are tuned for from-scratch ID
    # training (constantwithwarmup, 5000 warmup steps). For a short warm-
    # start fine-tune those defaults waste most of the budget on warmup and
    # never decay, so we override explicitly here.
    total_steps_estimate = max(1, total_tokens // batch_size)
    ft_warm_up_steps = max(1, int(0.05 * total_steps_estimate))
    ft_scheduler_name = "linearwarmupdecay"

    print(
        f"[OOD-Finetuned] Kissane-recipe fine-tune: total_tokens={total_tokens}, "
        f"lr={lr}, lambda_sparse={lambda_sparse}, batch_size={batch_size}, "
        f"warm_start=True, reinit_b_dec=False, loss_type={loss_type}, "
        f"scheduler={ft_scheduler_name}, warmup_steps={ft_warm_up_steps}/{total_steps_estimate}"
    )

    return evaluate_baseline_ood_retrained(
        model=model,
        ood_activations=ood_activations,
        ood_logits=ood_logits,
        layer=layer,
        model_type=model_type,
        d_model=d_model,
        vocab_size=vocab_size,
        # Training budget
        max_epochs=1,
        max_steps=None,
        total_tokens=total_tokens,
        lr=lr,
        lambda_sparse=lambda_sparse,
        objective_type="ERM",
        term_t=None,
        batch_size=batch_size,
        # Streaming path is required for the Kissane recipe
        use_streaming=True,
        ood_dataset=ood_dataset,
        ood_text_field=ood_text_field,
        ood_dataset_path=ood_dataset_path,
        ood_dataset_name=ood_dataset_name,
        context_size=context_size,
        input_hook_point=input_hook_point,
        output_hook_point=output_hook_point,
        # Warm-start enforced
        warm_start=True,
        reinit_b_dec=False,
        warm_start_explainer=warm_start_explainer,
        warm_start_path=warm_start_path,
        use_saelens=True,
        device=device,
        use_wandb=use_wandb,
        output_kind=output_kind,
        # Separate namespace so checkpoints don't collide with ood_retrained
        task_name=f"{task_name}_finetuned",
        ood_set=ood_set,
        base_checkpoint_path=base_checkpoint_path,
        retrained_checkpoint_path=finetuned_checkpoint_path,
        retrained_checkpoint_dir=finetuned_checkpoint_dir,
        # Kissane recipe LR schedule overrides
        lr_scheduler_name=ft_scheduler_name,
        lr_warm_up_steps=ft_warm_up_steps,
    )


def evaluate_baseline_term(
    checkpoint_path,
    model_type,
    d_model,
    vocab_size=None
):
    """
    Evaluate TERM baseline: load the TERM-trained explainer checkpoint and
    evaluate it with the same fixed-protocol used for other frozen explainers.
    
    Args:
        checkpoint_path: Path to explainer checkpoint to evaluate as TERM
        model_type: 'transcoder' or 'sae'
        d_model: Hidden dimension
        vocab_size: Vocabulary size (for Transcoder)
    
    Returns:
        explainer: Loaded explainer model
    """
    print(f"Loading TERM explainer from checkpoint: {checkpoint_path}")
    explainer = load_explainer(checkpoint_path, model_type, d_model, vocab_size)
    return explainer


def evaluate_baseline_faithfulsae(
    checkpoint_path,
    model_type,
    d_model,
    vocab_size=None,
):
    """Evaluate FaithfulSAE baseline: load explainer trained on model's own synthetic outputs.

    FaithfulSAE (arXiv:2506.17673) trains explainers on the model's own
    unconditional generation (from BOS token, temperature=1.0, top_p=0.9,
    repetition_penalty=1.1) instead of external datasets. This eliminates
    fake features caused by training data distribution mismatch.

    The loaded explainer is used as-is (no test-time adaptation), same as TERM.
    """
    print(f"Loading FaithfulSAE explainer from checkpoint: {checkpoint_path}")
    explainer = load_explainer(checkpoint_path, model_type, d_model, vocab_size)
    return explainer


def _compute_gae_ce_weights(model, eval_tokens_list, layer, explainer, hook_site_mode="auto"):
    """Compute CE-sensitivity weights for GAE affine decoder, if enabled."""
    if not getattr(Config, "GAE_CE_WEIGHTS", False):
        return None
    from gae import compute_ce_sensitivity_weights
    from ood_utils.evaluation import _resolve_faith_hook_site
    hook_site = _resolve_faith_hook_site(explainer, hook_site_mode=hook_site_mode)
    n_ce = int(getattr(Config, "GAE_CE_WEIGHTS_N", 512))
    tokens_subset = eval_tokens_list[:n_ce]
    eps = float(getattr(Config, "GAE_CE_WEIGHTS_EPS", 1e-8))
    print(f"[GAE] Computing CE-sensitivity weights (n={len(tokens_subset)}, hook={hook_site})")
    w = compute_ce_sensitivity_weights(
        model=model,
        tokens_list=tokens_subset,
        layer=layer,
        hook_site=hook_site,
        eps=eps,
    )
    print(f"[GAE] CE weights: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")
    return w


def evaluate_baseline_gae(
    explainer,
    model,
    id_activations,
    ood_activations,
    layer,
    model_type,
    r=64,
    device=None,
    id_activations_out=None,
    ood_activations_out=None,
    rank_mode="fixed",
    rank_energy=0.99,
    rank_delta_mode="ood",
    rank_min=1,
    rank_max=None,
    decoder_mode="refit",
    dict_space="auto",
    recon_lambda=0.0,
    ce_weights=None,
):
    """
    Evaluate GAE baseline: adapt ID-trained explainer using GAE.

    Args:
        decoder_mode:
            - 'theory': use B_GAE as decoder, preserve original encoder
            - 'affine': affine constrained decoder fit with GAE geometry prior
              (generalized form; theory is its limit case with lam_gae→∞,
              lam_id=lam_geom=0, frozen bias)
        dict_space: 'auto'/'decoder' (use W_dec), 'empirical' (legacy h-space LS).

    NOTE (2026-04-14): Removed decoder modes [sc, diag, mix, consistent, none, refit].
    These were previously implemented but consistently underperformed across all
    OOD settings. See docs/gae_260414.md (Section: Removed decoder modes).
    """
    if decoder_mode not in ("theory", "affine", "residual_orig", "residual_rotated",
                            "encoder_rotation", "selective_columns", "diag_gains"):
        raise ValueError(
            f"Unknown decoder_mode='{decoder_mode}'. "
            "Supported: theory, affine, residual_orig, residual_rotated, "
            "encoder_rotation, selective_columns, diag_gains."
        )
    if device is None:
        device = Config.device
    
    if rank_mode not in ("fixed", "energy"):
        raise ValueError(f"Unknown rank_mode='{rank_mode}'. Use 'fixed' or 'energy'.")
    if rank_delta_mode not in ("ood", "contrastive"):
        raise ValueError(f"Unknown rank_delta_mode='{rank_delta_mode}'. Use 'ood' or 'contrastive'.")
    if not (0.0 < float(rank_energy) <= 1.0):
        raise ValueError(f"rank_energy must be in (0, 1], got {rank_energy}")
    if int(rank_min) <= 0:
        raise ValueError("rank_min must be positive.")
    if rank_max is not None and int(rank_max) <= 0:
        raise ValueError("rank_max must be positive when provided.")

    gae_geometry = _resolve_gae_geometry_activations(
        explainer=explainer,
        model_type=model_type,
        dict_space=dict_space,
        id_activations=id_activations,
        ood_activations=ood_activations,
        id_activations_out=id_activations_out,
        ood_activations_out=ood_activations_out,
    )

    print(f"Applying GAE to explainer (rank_mode={rank_mode}, r={r})...")
    print(
        "[GAE] Geometry space: "
        f"{gae_geometry['space']} (dict_space={gae_geometry['dict_space']}, source={gae_geometry['source']})"
    )
    with tqdm(total=6, desc="GAE", leave=True) as pbar:
        # 1) Extract original dictionary
        B_original = extract_dictionary_from_explainer(explainer, id_activations, model_type, dict_space=dict_space)
        d, k = B_original.shape
        pbar.update(1)

        device_local = B_original.device
        ood_activations_local = ood_activations.to(device_local)
        id_activations_local = id_activations.to(device_local)
        gae_ood_activations_local = gae_geometry["ood"].to(device_local)
        gae_id_activations_local = gae_geometry["id"].to(device_local)

        # 2) Select rank (fixed or energy-based)
        if rank_mode == "fixed":
            r_eff = min(int(r), min(d, k))
            rank_info = None
        else:
            r_eff, rank_info = select_rank_for_gae(
                B=B_original,
                H_ood=gae_ood_activations_local,
                H_id=gae_id_activations_local,
                energy_ratio=float(rank_energy),
                delta_mode=rank_delta_mode,
                min_rank=int(rank_min),
                max_rank=int(rank_max) if rank_max is not None else None,
            )
            r_eff = int(r_eff)
            print(
                f"[GAE] Energy rank selected: r={r_eff}, "
                f"captured={rank_info['energy_ratio_captured']:.6f}, "
                f"target={rank_info['energy_ratio_target']:.6f}, "
                f"cap={rank_info['rank_cap']}, delta_mode={rank_info['delta_mode']}"
            )
        pbar.update(1)

        # 3) Compute GAE dictionary
        print(f"Applying GAE with rank r={r_eff} (d={d}, k={k})")
        print(f"  ID activations (geometry): {gae_geometry['id'].shape}")
        print(f"  OOD activations (geometry): {gae_geometry['ood'].shape}")

        _Z_ood = None
        _H_target_recon = None
        _b_target_recon = None
        if float(recon_lambda) > 0.0:
            with torch.no_grad():
                _is_transcoder = False
                if hasattr(explainer, "saelens"):
                    _sae = explainer.saelens
                    _is_transcoder = bool(getattr(getattr(_sae, "cfg", None), "is_transcoder", False))
                    if _is_transcoder and hasattr(_sae, "b_dec_out"):
                        _b_target_recon = _sae.b_dec_out.detach().to(device_local)
                    elif hasattr(_sae, "b_dec"):
                        _b_target_recon = _sae.b_dec.detach().to(device_local)
                elif hasattr(explainer, "b_dec"):
                    _b_target_recon = explainer.b_dec.detach().to(device_local)

                if _is_transcoder and ood_activations_out is not None:
                    _H_target_recon = ood_activations_out.to(device_local)
                    if _H_target_recon.shape[0] != gae_ood_activations_local.shape[0]:
                        _n_align = min(_H_target_recon.shape[0], gae_ood_activations_local.shape[0])
                        _H_target_recon = _H_target_recon[:_n_align]
                else:
                    _H_target_recon = gae_ood_activations_local

                _z_batches = []
                _CHUNK_ENC = 4096
                _N_enc = _H_target_recon.shape[0]
                for _s in range(0, _N_enc, _CHUNK_ENC):
                    _e = min(_s + _CHUNK_ENC, _N_enc)
                    _, _z = explainer(gae_ood_activations_local[_s:_e], return_z=True)
                    _z_batches.append(_z.detach().to(device_local))
                _Z_ood = torch.cat(_z_batches, dim=0)
                del _z_batches

                if _b_target_recon is None:
                    _b_target_recon = torch.zeros(d, device=device_local, dtype=B_original.dtype)

                assert _Z_ood.shape[0] == _H_target_recon.shape[0], (
                    f"N mismatch: Z={tuple(_Z_ood.shape)}, "
                    f"H_target={tuple(_H_target_recon.shape)}"
                )
                print(
                    f"[GAE] Recon term enabled: lambda={float(recon_lambda)}, "
                    f"N={_Z_ood.shape[0]}, is_transcoder={_is_transcoder}"
                )

        B_gae, gae_info = gae_torch(
            B_original,
            gae_ood_activations_local,
            r_eff,
            recon_lambda=float(recon_lambda),
            Z=_Z_ood,
            H_target=_H_target_recon,
            b_target=_b_target_recon,
        )
        del _Z_ood, _H_target_recon, _b_target_recon
        pbar.update(1)

        # 4) Match dictionary scale
        with torch.no_grad():
            orig_norms = B_original.norm(dim=0)
            gae_norms = B_gae.norm(dim=0)
            scale = orig_norms / (gae_norms + 1e-8)
            B_gae = B_gae * scale.unsqueeze(0)
        pbar.update(1)

        # 5) Decoder / feature-adapter selection
        D_star = None
        feature_adapter = None
        feature_gains = None
        extra_metadata = {}
        decoder_bias = None
        if hasattr(explainer, "saelens"):
            sae = explainer.saelens
            if getattr(sae.cfg, "is_transcoder", False) and hasattr(sae, "b_dec_out"):
                decoder_bias = sae.b_dec_out.detach().clone()
            elif hasattr(sae, "b_dec"):
                decoder_bias = sae.b_dec.detach().clone()
        elif hasattr(explainer, "b_dec"):
            decoder_bias = explainer.b_dec.detach().clone()

        # NOTE (2026-04-14): Removed decoder modes [sc, diag, mix, consistent, none, refit].
        # These were previously implemented but consistently underperformed compared to
        # `theory` and `affine` across all OOD settings during HP search. Documented in
        # docs/gae_260414.md (Section: Removed decoder modes). Code preserved in commit
        # history if needed.
        if decoder_mode == "theory":
            # Theory-aligned: B_GAE used as decoder in evaluation (no separate D_star)
            print("[GAE] Theory mode: B_GAE will be used as decoder (original encoder preserved)")
            pbar.update(1)
        elif decoder_mode == "affine":
            target_for_affine = (
                ood_activations_out.to(device_local)
                if ood_activations_out is not None
                else ood_activations_local
            )
            D_star, decoder_bias, affine_info = fit_affine_constrained_decoder(
                explainer=explainer,
                activations_in=ood_activations_local,
                activations_target=target_for_affine,
                B_original=B_original,
                B_gae=B_gae,
                U_ood=gae_info["U_ood"],
                lam_geom=float(getattr(Config, "GAE_AC_LAMBDA_GEOM", 1e-1)),
                lam_id=float(getattr(Config, "GAE_AC_LAMBDA_ID", 5e-2)),
                lam_gae=float(getattr(Config, "GAE_AC_LAMBDA_GAE", 2e-1)),
                fit_samples=int(getattr(Config, "GAE_AC_FIT_SAMPLES", 2048)),
                batch_size=int(getattr(Config, "GAE_AC_BATCH_SIZE", 128)),
                solver_device=str(getattr(Config, "GAE_AC_SOLVER_DEVICE", "cpu")),
                match_column_norms=bool(getattr(Config, "GAE_AC_MATCH_COLUMN_NORMS", False)),
                decoder_mix_prior=float(getattr(Config, "GAE_AC_DECODER_MIX_PRIOR", 0.0)),
                ce_weights=ce_weights,
            )
            decoder_mix_orig = float(getattr(Config, "GAE_AC_DECODER_MIX_ORIG", 0.0))
            if D_star is not None and decoder_mix_orig > 0.0:
                decoder_mix_orig = max(0.0, min(1.0, decoder_mix_orig))
                B_orig_mix = B_original.to(device=D_star.device, dtype=D_star.dtype)
                D_star = (1.0 - decoder_mix_orig) * D_star + decoder_mix_orig * B_orig_mix
                affine_info["decoder_mix_orig"] = float(decoder_mix_orig)
            bias_mix_orig = float(getattr(Config, "GAE_AC_BIAS_MIX_ORIG", 0.0))
            if decoder_bias is not None and decoder_bias is not None and bias_mix_orig > 0.0:
                bias_mix_orig = max(0.0, min(1.0, bias_mix_orig))
                if hasattr(explainer, "saelens"):
                    sae = explainer.saelens
                    if getattr(sae.cfg, "is_transcoder", False) and hasattr(sae, "b_dec_out"):
                        orig_bias = sae.b_dec_out.detach().clone().to(decoder_bias.device, dtype=decoder_bias.dtype)
                    elif hasattr(sae, "b_dec"):
                        orig_bias = sae.b_dec.detach().clone().to(decoder_bias.device, dtype=decoder_bias.dtype)
                    else:
                        orig_bias = None
                elif hasattr(explainer, "b_dec"):
                    orig_bias = explainer.b_dec.detach().clone().to(decoder_bias.device, dtype=decoder_bias.dtype)
                else:
                    orig_bias = None
                if orig_bias is not None:
                    decoder_bias = (1.0 - bias_mix_orig) * decoder_bias + bias_mix_orig * orig_bias
                    affine_info["bias_mix_orig"] = float(bias_mix_orig)
            extra_metadata["affine_constrained"] = affine_info
            print(
                "[GAE] Affine constrained decoder fit enabled "
                f"(fit_mse={affine_info['fit_mse']:.4f}, n={affine_info['fit_samples']}, rank={affine_info['rank']})"
            )
            pbar.update(1)
        elif decoder_mode in ("residual_orig", "residual_rotated"):
            from gae import fit_residual_decoder
            lam_res = float(getattr(Config, "GAE_AC_LAMBDA_RES", 0.1))

            target_for_residual = (
                ood_activations_out.to(device_local)
                if ood_activations_out is not None
                else ood_activations_local
            )

            # Collect Z features from frozen encoder
            z_batches = []
            batch_sz = int(getattr(Config, "GAE_AC_BATCH_SIZE", 128))
            for start in range(0, int(ood_activations_local.shape[0]), batch_sz):
                end = min(int(ood_activations_local.shape[0]), start + batch_sz)
                _, z_b = explainer(ood_activations_local[start:end], return_z=True)
                z_batches.append(z_b.detach().to(device="cpu", dtype=torch.float32))
            Z_ood = torch.cat(z_batches, dim=0)
            H_ood = target_for_residual.to(device="cpu", dtype=torch.float32)

            if decoder_mode == "residual_rotated":
                # Base = Step 1 rotated decoder W̃
                W_base = B_gae.to(device="cpu", dtype=torch.float32)
                # Compute rotated bias: b̃ = mean(h) - W̃ @ mean(z)
                b_base = H_ood.mean(0) - W_base @ Z_ood.mean(0)
            else:
                # Base = original ID decoder W_orig
                W_base = B_original.to(device="cpu", dtype=torch.float32)
                # Original decoder bias
                if hasattr(explainer, "saelens"):
                    sae = explainer.saelens
                    if getattr(sae.cfg, "is_transcoder", False) and hasattr(sae, "b_dec_out"):
                        b_base = sae.b_dec_out.detach().clone().to(device="cpu", dtype=torch.float32)
                    elif hasattr(sae, "b_dec"):
                        b_base = sae.b_dec.detach().clone().to(device="cpu", dtype=torch.float32)
                    else:
                        b_base = torch.zeros(W_base.shape[0], dtype=torch.float32)
                elif hasattr(explainer, "b_dec"):
                    b_base = explainer.b_dec.detach().clone().to(device="cpu", dtype=torch.float32)
                else:
                    b_base = torch.zeros(W_base.shape[0], dtype=torch.float32)

            D_star, decoder_bias, residual_info = fit_residual_decoder(
                W_base, b_base, H_ood, Z_ood, lam_res=lam_res,
            )
            extra_metadata["residual_fit"] = residual_info
            print(
                f"[GAE] Residual decoder fit ({decoder_mode}): "
                f"base_mse={residual_info['base_mse']:.4f} -> fit_mse={residual_info['fit_mse']:.4f}, "
                f"dW_ratio={residual_info['delta_W_ratio']:.4f}, lam_res={lam_res}"
            )
            pbar.update(1)
        elif decoder_mode == "encoder_rotation":
            # SAE-only: apply Procrustes rotation R* (computed from gae_torch above)
            # to encoder weights. Decoder (W_dec, b_dec) is preserved.
            if model_type != "sae":
                raise ValueError("decoder_mode='encoder_rotation' is SAE-only.")
            if not hasattr(explainer, "saelens"):
                raise ValueError("encoder_rotation requires SaeLensSparseAutoencoderAdapter.")
            from gae import apply_gae_encoder_rotation
            enc_mix = float(getattr(Config, "GAE_SAE_ENCODER_MIX", 1.0))
            enc_preserve = bool(int(getattr(Config, "GAE_SAE_ENCODER_PRESERVE_ORTHOGONAL", 1)))
            enc_info = apply_gae_encoder_rotation(
                explainer.saelens,
                U_ood=gae_info["U_ood"],
                U_ref=gae_info["U_ref"],
                R_align=gae_info["R_align"],
                preserve_orthogonal=enc_preserve,
                mix=enc_mix,
            )
            extra_metadata["encoder_rotation"] = enc_info
            print(
                f"[GAE] Encoder rotation applied: mix={enc_info['mix']}, "
                f"preserve_ortho={enc_info['preserve_orthogonal']}, "
                f"delta_ratio={enc_info['delta_ratio']:.4f}"
            )
            # No decoder override: D_star and decoder_bias remain None (uses original SAE decoder).
            pbar.update(1)
        elif decoder_mode == "selective_columns":
            # SAE-only: update only OOD-active decoder columns; freeze others at W_dec.
            if model_type != "sae":
                raise ValueError("decoder_mode='selective_columns' is SAE-only.")
            if not hasattr(explainer, "saelens"):
                raise ValueError("selective_columns requires SaeLensSparseAutoencoderAdapter.")
            from gae import fit_selective_decoder_columns
            sae = explainer.saelens
            W_dec_orig = sae.W_dec.detach().clone().to(device="cpu", dtype=torch.float32)  # [k, d]
            b_dec_orig = sae.b_dec.detach().clone().to(device="cpu", dtype=torch.float32)  # [d]

            # Compute Z (encoder features) on OOD activations
            z_batches = []
            batch_sz = int(getattr(Config, "GAE_AC_BATCH_SIZE", 128))
            for start in range(0, int(ood_activations_local.shape[0]), batch_sz):
                end = min(int(ood_activations_local.shape[0]), start + batch_sz)
                _, z_b = explainer(ood_activations_local[start:end], return_z=True)
                z_batches.append(z_b.detach().to(device="cpu", dtype=torch.float32))
            Z_ood = torch.cat(z_batches, dim=0)  # [N, k]
            H_ood = ood_activations_local.to(device="cpu", dtype=torch.float32)

            # Optionally compute Z_id to find OOD-only active features
            ood_only_mode = str(getattr(Config, "GAE_SAE_SEL_MODE", "ood_active"))  # "ood_active" or "ood_only"
            ood_active_mask = (Z_ood > 0).any(dim=0)  # [k] features active in OOD
            if ood_only_mode == "ood_only":
                # Compute Z on ID activations
                z_id_batches = []
                for start in range(0, int(id_activations_local.shape[0]), batch_sz):
                    end = min(int(id_activations_local.shape[0]), start + batch_sz)
                    _, z_b = explainer(id_activations_local[start:end], return_z=True)
                    z_id_batches.append(z_b.detach().to(device="cpu", dtype=torch.float32))
                Z_id = torch.cat(z_id_batches, dim=0)
                id_active_mask = (Z_id > 0).any(dim=0)
                # OOD-only active = OOD active AND NOT ID active
                col_mask = ood_active_mask & ~id_active_mask
            else:
                # All OOD-active features
                col_mask = ood_active_mask

            column_indices = col_mask.nonzero(as_tuple=False).squeeze(-1)  # [n_sel]
            n_sel = int(column_indices.numel())
            print(f"[GAE] selective_columns: updating {n_sel}/{Z_ood.shape[1]} columns "
                  f"(mode={ood_only_mode})")
            if n_sel == 0:
                print("[GAE] No columns to update; falling back to Fixed (D=W_dec, b=b_dec)")
                D_star = None
                decoder_bias = None
            else:
                lam_res = float(getattr(Config, "GAE_AC_LAMBDA_RES", 0.1))
                W_new, b_new, sel_info = fit_selective_decoder_columns(
                    W_dec_orig, b_dec_orig, H_ood, Z_ood,
                    column_indices=column_indices, lam_res=lam_res,
                )
                D_star = W_new.T  # [d, k] (run_experiment convention is decoder_override [d, k])
                decoder_bias = b_new
                extra_metadata["selective_fit"] = sel_info
                print(f"[GAE] selective fit: base_mse={sel_info['base_mse']:.4f} -> "
                      f"fit_mse={sel_info['fit_mse']:.4f}, lam_res={lam_res}, "
                      f"frac_updated={sel_info['frac_columns_updated']:.4f}")
            pbar.update(1)
        elif decoder_mode == "diag_gains":
            # SAE-only: per-column magnitude scaling of decoder (direction preserved).
            # Equivalent to per-feature gains: z_eff = z * α → h_recon = z_eff @ W_dec + b.
            # Special case of Step 2 affine fit with diagonal constraint.
            from gae import fit_diag_feature_gains
            target_for_diag = (
                ood_activations_out.to(device_local)
                if ood_activations_out is not None
                else ood_activations_local
            )
            diag_lam = float(getattr(Config, "GAE_AC_DIAG_LAM", 1e-3))
            gains = fit_diag_feature_gains(
                explainer, ood_activations_local, target_for_diag,
                decoder=B_original,  # [d, k]
                lam=diag_lam,
                batch_size=int(getattr(Config, "GAE_AC_BATCH_SIZE", 128)),
            )
            # Clamp gains for stability (prevent extreme scaling)
            gains_min = float(getattr(Config, "GAE_AC_DIAG_CLAMP_MIN", 0.5))
            gains_max = float(getattr(Config, "GAE_AC_DIAG_CLAMP_MAX", 2.0))
            gains_clamped = gains.clamp(min=gains_min, max=gains_max)
            # Apply as decoder column scaling: D_star[:, j] = gains[j] * B_original[:, j]
            D_star = B_original.clone() * gains_clamped.unsqueeze(0)  # [d, k]
            decoder_bias = None  # use original b_dec
            extra_metadata["diag_gains"] = {
                "diag_lam": diag_lam,
                "gains_raw_min": float(gains.min().item()),
                "gains_raw_max": float(gains.max().item()),
                "gains_raw_mean": float(gains.mean().item()),
                "gains_clamped_min": float(gains_min),
                "gains_clamped_max": float(gains_max),
                "n_clamped_low": int((gains < gains_min).sum().item()),
                "n_clamped_high": int((gains > gains_max).sum().item()),
            }
            print(f"[GAE] Diagonal gains: lam={diag_lam}, gains [{gains.min():.3f}, {gains.max():.3f}] "
                  f"→ clamped [{gains_min}, {gains_max}], "
                  f"n_clamp_lo={int((gains < gains_min).sum())}, n_clamp_hi={int((gains > gains_max).sum())}")
            pbar.update(1)
        else:
            raise ValueError(
                f"Unsupported decoder_mode={decoder_mode!r}."
            )

        # 6) Done
        pbar.update(1)

    print(f"\nGAE Sanity Checks:")
    print(f"  B_original shape: {B_original.shape}")
    print(f"  B_gae shape: {B_gae.shape}")
    assert B_gae.shape == B_original.shape, "Shape mismatch after GAE"

    # Store GAE components on explainer so evaluation/diagnostics can use them
    target_device = explainer.W_tilde.device if hasattr(explainer, "W_tilde") else B_gae.device
    explainer.B_gae = B_gae.to(target_device)  # [d_model, k]
    if decoder_mode in ("theory", "affine", "residual_orig", "residual_rotated",
                        "encoder_rotation", "selective_columns", "diag_gains"):
        explainer.gae_apply_to = "decoder"
        if hasattr(explainer, "D_gae"):
            delattr(explainer, "D_gae")
        if D_star is not None:
            explainer.gae_decoder_override = D_star.to(target_device)
        elif hasattr(explainer, "gae_decoder_override"):
            delattr(explainer, "gae_decoder_override")
        if decoder_bias is not None:
            explainer.gae_decoder_bias = decoder_bias.to(target_device)
            print(f"[GAE] Decoder bias preserved (norm={decoder_bias.norm():.4f})")
        else:
            explainer.gae_decoder_bias = None
        explainer.gae_feature_adapter = feature_adapter
        if feature_gains is not None:
            explainer.gae_feature_gains = feature_gains.to(target_device)
        elif hasattr(explainer, "gae_feature_gains"):
            delattr(explainer, "gae_feature_gains")
        explainer.gae_support_refit_ridge = float(getattr(Config, "GAE_SUPPORT_REFIT_RIDGE", 1e-3))
        explainer.gae_support_refit_topk = int(getattr(Config, "GAE_SUPPORT_REFIT_TOPK", 128))
    else:
        # Legacy modes: B_GAE as encoder
        explainer.gae_apply_to = "encoder"
        if hasattr(explainer, "gae_decoder_override"):
            delattr(explainer, "gae_decoder_override")
        explainer.gae_decoder_bias = None
        explainer.gae_feature_adapter = None
        if hasattr(explainer, "gae_feature_gains"):
            delattr(explainer, "gae_feature_gains")
        if D_star is not None:
            explainer.D_gae = D_star.to(target_device)  # [d_model, k]
        elif hasattr(explainer, "D_gae"):
            delattr(explainer, "D_gae")
    explainer.use_gae = True
    explainer.gae_decoder_mode = decoder_mode
    explainer.gae_rank_mode = rank_mode
    explainer.gae_rank_eff = int(r_eff)
    explainer.gae_geometry_space = gae_geometry["space"]
    explainer.gae_geometry_source = gae_geometry["source"]
    explainer.gae_dict_space = gae_geometry["dict_space"]
    explainer.gae_extra_metadata = extra_metadata
    explainer.gae_recon_lambda = float(recon_lambda)
    explainer.gae_procrustes_convention = gae_info.get("procrustes_convention", "UV^T")
    if "tau_rec" in gae_info:
        explainer.gae_tau_rec = float(gae_info["tau_rec"])
        explainer.gae_M_rec_frobenius = float(gae_info.get("M_rec_frobenius", 0.0))
    if rank_mode == "energy":
        explainer.gae_rank_info = rank_info

    return explainer


def _resolve_saeboost_residual_checkpoint(
    task_name: str,
    model_name: str,
    explainer_type: str,
    ood_set: str,
    base_checkpoint_path: Optional[str] = None,
    residual_checkpoint_path: Optional[str] = None,
    residual_checkpoint_dir: str = "checkpoints",
) -> str:
    """
    Resolve residual checkpoint path for SAEBoost.
    Priority:
      1) Explicit --saeboost_residual_checkpoint
      2) Suggested naming under --saeboost_residual_dir
      3) Suggested sibling paths near base checkpoint
    """
    candidates = []
    expected_name = _saeboost_residual_expected_name(model_name, explainer_type, ood_set)

    if residual_checkpoint_path is not None:
        candidates.append(Path(residual_checkpoint_path))

    if residual_checkpoint_dir:
        root = Path(residual_checkpoint_dir)
        candidates.append(root / task_name / expected_name)
        candidates.append(root / f"{task_name}_{expected_name}")
        candidates.append(root / expected_name)

    if base_checkpoint_path:
        base_parent = Path(base_checkpoint_path).resolve().parent
        candidates.append(base_parent / expected_name)
        candidates.append(base_parent / task_name / expected_name)

    seen = set()
    ordered = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(c)

    for c in ordered:
        if c.exists() and c.is_file():
            return str(c)

    if residual_checkpoint_path:
        raise FileNotFoundError(
            f"SAEBoost residual checkpoint not found at '{residual_checkpoint_path}'."
        )

    searched = "\n".join(f"  - {p}" for p in ordered) if ordered else "  (no candidates)"
    raise FileNotFoundError(
        "Could not locate SAEBoost residual checkpoint.\n"
        f"Searched:\n{searched}\n"
        "Provide --saeboost_residual_checkpoint explicitly to override."
    )


def evaluate_baseline_saeboost(
    base_explainer,
    model: Optional[HookedTransformer],
    layer: Optional[int],
    ood_dataset,
    ood_text_field,
    model_type: str,
    d_model: int,
    vocab_size: Optional[int],
    task_name: str,
    model_name: str,
    ood_set: str,
    base_checkpoint_path: Optional[str] = None,
    residual_checkpoint_path: Optional[str] = None,
    residual_checkpoint_dir: str = "checkpoints",
    resid_coef: float = 1.0,
    train_residual_if_missing: bool = False,
    residual_train_max_epochs: int = 50,
    residual_train_max_steps: Optional[int] = None,
    residual_train_total_tokens: Optional[int] = None,
    residual_train_lr: float = 8e-4,
    residual_train_lambda_sparse: float = 5e-6,
    residual_train_objective: str = "ERM",
    residual_train_term_t: Optional[float] = None,
    residual_train_batch_size: Optional[int] = None,
    residual_train_context_size: int = 128,
    residual_train_dict_size: Optional[int] = 1024,
    residual_train_reinit_b_dec: bool = True,
    residual_train_sparse_method: str = "l1",
    residual_train_batch_top_k: int = 5,
    residual_train_batch_top_k_start: Optional[int] = None,
    residual_train_batch_top_k_warmup_fraction: float = 0.1,
    base_inference_sparse_method: Optional[str] = None,
    base_inference_batch_top_k: Optional[int] = None,
    base_jumprelu_threshold: Optional[float] = None,
    residual_inference_sparse_method: Optional[str] = "auto",
    residual_inference_batch_top_k: Optional[int] = None,
    residual_jumprelu_threshold: Optional[float] = None,
    use_wandb: bool = False,
):
    """
    Build SAEBoost baseline explainer from base(ID) + residual(OOD) checkpoints.
    """
    if float(resid_coef) == 0.0:
        print("[SAEBoost] resid_coef=0.0 -> using base explainer directly (exact fixed-equivalent path).")
        return base_explainer

    resid_path = None
    residual_explainer = None

    try:
        resid_path = _resolve_saeboost_residual_checkpoint(
            task_name=task_name,
            model_name=model_name,
            explainer_type=model_type,
            ood_set=ood_set,
            base_checkpoint_path=base_checkpoint_path,
            residual_checkpoint_path=residual_checkpoint_path,
            residual_checkpoint_dir=residual_checkpoint_dir,
        )
    except FileNotFoundError as err:
        if not train_residual_if_missing:
            raise
        print(f"[SAEBoost] Residual checkpoint not found. Training a new one. ({err})")

    if resid_path is None:
        resid_path = (
            residual_checkpoint_path
            if residual_checkpoint_path is not None
            else _saeboost_default_residual_checkpoint_path(
                task_name=task_name,
                model_name=model_name,
                explainer_type=model_type,
                ood_set=ood_set,
                residual_checkpoint_dir=residual_checkpoint_dir,
            )
        )
        if model is None or layer is None or ood_dataset is None or ood_text_field is None:
            raise ValueError(
                "SAEBoost residual training requires model/layer/ood_dataset/ood_text_field."
            )
        train_batch_size = residual_train_batch_size or Config.BATCH_SIZE
        print(
            "[SAEBoost] Training residual checkpoint "
            f"(save_path={resid_path}, lr={residual_train_lr}, "
            f"lambda_sparse={residual_train_lambda_sparse}, "
            f"dict_size={residual_train_dict_size}, total_tokens={residual_train_total_tokens})"
        )
        residual_saelens = _train_saeboost_residual_saelens_streaming(
            base_explainer=base_explainer,
            model=model,
            dataset=ood_dataset,
            text_field=ood_text_field,
            layer=layer,
            context_size=residual_train_context_size,
            batch_size=train_batch_size,
            max_epochs=residual_train_max_epochs,
            max_steps=residual_train_max_steps,
            total_tokens=residual_train_total_tokens,
            lr=residual_train_lr,
            lambda_sparse=residual_train_lambda_sparse,
            objective_type=residual_train_objective,
            term_t=residual_train_term_t,
            residual_dict_size=residual_train_dict_size,
            reinit_b_dec=residual_train_reinit_b_dec,
            sparse_method=residual_train_sparse_method,
            residual_batch_top_k=residual_train_batch_top_k,
            residual_batch_top_k_start=residual_train_batch_top_k_start,
            residual_batch_top_k_warmup_fraction=residual_train_batch_top_k_warmup_fraction,
            device=Config.device,
            use_wandb=use_wandb,
            wandb_prefix="saeboost_residual",
        )
        _save_saelens_checkpoint(residual_saelens, resid_path)
        if model_type == "transcoder":
            residual_explainer = SaeLensTranscoderAdapter(residual_saelens)
        else:
            residual_explainer = SaeLensSparseAutoencoderAdapter(residual_saelens)
    else:
        print(f"Loading SAEBoost residual checkpoint: {resid_path}")
        residual_explainer = load_explainer(
            resid_path,
            model_type=model_type,
            d_model=d_model,
            vocab_size=vocab_size,
        )

    residual_cfg = getattr(getattr(residual_explainer, "saelens", None), "cfg", None)
    resolved_resid_sparse_method = residual_inference_sparse_method
    if resolved_resid_sparse_method is None or str(resolved_resid_sparse_method).lower() == "auto":
        inferred_sparse_method = None
        if residual_cfg is not None:
            inferred_sparse_method = getattr(residual_cfg, "saeboost_inference_sparse_method", None)
            if inferred_sparse_method is None:
                inferred_sparse_method = getattr(residual_cfg, "saeboost_sparse_method", None)
        if inferred_sparse_method is None and str(residual_train_sparse_method).lower() == "batchtopk":
            inferred_sparse_method = "batchtopk"
        resolved_resid_sparse_method = inferred_sparse_method if inferred_sparse_method is not None else "auto"

    resolved_resid_batch_top_k = residual_inference_batch_top_k
    if resolved_resid_batch_top_k is None:
        inferred_batch_top_k = None
        if residual_cfg is not None:
            inferred_batch_top_k = getattr(residual_cfg, "saeboost_batch_top_k", None)
            if inferred_batch_top_k is None:
                inferred_batch_top_k = getattr(residual_cfg, "top_k", None)
        if inferred_batch_top_k is None and str(resolved_resid_sparse_method).lower() == "batchtopk":
            inferred_batch_top_k = residual_train_batch_top_k
        if inferred_batch_top_k is not None:
            resolved_resid_batch_top_k = int(inferred_batch_top_k)

    resolved_resid_jumprelu_threshold = residual_jumprelu_threshold
    if resolved_resid_jumprelu_threshold is None and residual_cfg is not None:
        cfg_threshold = getattr(residual_cfg, "saeboost_jumprelu_threshold", None)
        if cfg_threshold is not None:
            resolved_resid_jumprelu_threshold = float(cfg_threshold)

    if model_type == "transcoder":
        if getattr(base_explainer, "output_kind", None) != "mlp_out":
            raise NotImplementedError(
                "SAEBoost transcoder is only supported for base output_kind='mlp_out'."
            )
        if getattr(residual_explainer, "output_kind", None) != "mlp_out":
            raise NotImplementedError(
                "SAEBoost transcoder requires residual checkpoint with output_kind='mlp_out'."
            )
        return SAEBoostTranscoderAdapter(
            base_transcoder=base_explainer,
            resid_transcoder=residual_explainer,
            resid_coef=resid_coef,
            base_sparse_method=base_inference_sparse_method,
            base_batch_top_k=base_inference_batch_top_k,
            base_jumprelu_threshold=base_jumprelu_threshold,
            resid_sparse_method=resolved_resid_sparse_method,
            resid_batch_top_k=resolved_resid_batch_top_k,
            resid_jumprelu_threshold=resolved_resid_jumprelu_threshold,
        )

    if model_type == "sae":
        return SAEBoostExplainerAdapter(
            base_explainer=base_explainer,
            resid_explainer=residual_explainer,
            resid_coef=resid_coef,
            base_sparse_method=base_inference_sparse_method,
            base_batch_top_k=base_inference_batch_top_k,
            base_jumprelu_threshold=base_jumprelu_threshold,
            resid_sparse_method=resolved_resid_sparse_method,
            resid_batch_top_k=resolved_resid_batch_top_k,
            resid_jumprelu_threshold=resolved_resid_jumprelu_threshold,
        )

    raise ValueError(f"Unsupported explainer type for SAEBoost: {model_type}")


def run_adversarial_ood_experiment(
    model_name,
    explainer_type,
    ood_set,
    explainer_checkpoint_path,
    output_dir=None,
    n_eval_samples=1000,
    force_recompute_activations=False,
    use_wandb=False,
    wandb_project='GAE_Adversarial-OOD',
    wandb_name=None,
    baselines=None,
    # OOD-retrained hyperparameters
    ood_retrained_max_epochs=50,
    ood_retrained_max_steps=None,
    ood_retrained_total_tokens=None,
    ood_retrained_lr=1e-3,
    ood_retrained_lambda_sparse=0.01,
    ood_retrained_objective="ERM",
    ood_retrained_term_t=None,
    ood_retrained_batch_size=None,
    ood_retrained_streaming=False,
    ood_retrained_context_size=128,
    ood_retrained_warm_start=True,
    ood_retrained_reinit_b_dec=True,
    ood_retrained_use_saelens=True,
    ood_retrained_checkpoint_path=None,
    ood_retrained_checkpoint_dir=os.environ.get("REPO_DATA", "./data") + "/checkpoints",
    # OOD-finetuned hyperparameters
    ood_finetuned_total_tokens=5_000_000,
    ood_finetuned_lr=None,
    ood_finetuned_lambda_sparse=None,
    ood_finetuned_batch_size=256,
    ood_finetuned_context_size=128,
    ood_finetuned_loss_type="mse",
    ood_finetuned_checkpoint_path=None,
    ood_finetuned_checkpoint_dir=os.environ.get("REPO_DATA", "./data") + "/checkpoints/ood_finetuned",
    # TERM checkpoint
    term_checkpoint_path=None,
    # FaithfulSAE checkpoint
    faithfulsae_checkpoint_path=None,
    # GAE hyperparameters
    gae_r=64,
    gae_rank_mode="fixed",
    gae_rank_energy=0.99,
    gae_rank_delta_mode="ood",
    gae_rank_min=1,
    gae_rank_max=None,
    gae_decoder_mode="theory",
    gae_dict_space="auto",
    gae_recon_lambda=0.0,
    # SAEBoost checkpoint options
    saeboost_residual_checkpoint=None,
    saeboost_residual_dir="checkpoints",
    saeboost_coef=1.0,
    saeboost_train_residual=False,
    saeboost_residual_max_epochs=50,
    saeboost_residual_max_steps=None,
    saeboost_residual_total_tokens=None,
    saeboost_residual_lr=8e-4,
    saeboost_residual_lambda_sparse=5e-6,
    saeboost_residual_objective="ERM",
    saeboost_residual_term_t=None,
    saeboost_residual_batch_size=None,
    saeboost_residual_context_size=128,
    saeboost_residual_dict_size=1024,
    saeboost_residual_reinit_b_dec=True,
    saeboost_residual_sparse_method="l1",
    saeboost_residual_batch_top_k=5,
    saeboost_residual_batch_top_k_start=None,
    saeboost_residual_batch_top_k_warmup_fraction=0.1,
    saeboost_base_infer_sparse_method="auto",
    saeboost_base_infer_batch_top_k=None,
    saeboost_base_jumprelu_threshold=None,
    saeboost_residual_infer_sparse_method="auto",
    saeboost_residual_infer_batch_top_k=None,
    saeboost_residual_jumprelu_threshold=None,
    # Target mode and FT
    target_mode='argmax',
    compute_ft=True,
    min_len_ood_override=None,
    min_len_eval_override=None,
    adv_ood_token_budget=2_000_000,
    adv_bootstrap_repeats=100,
    adv_bootstrap_ci=95.0,
    faith_m_values=None,
    faith_random_repeats=5,
    faith_m_star=32,
):
    """
    Run Adversarial OOD experiment.

    Args:
        model_name: Model identifier ('gpt2', 'pythia-410m', 'pythia-1.4b')
        explainer_type: 'transcoder' or 'sae'
        ood_set: 'halu_eval', 'jailbreakbench', or 'jailbreakhub'
        explainer_checkpoint_path: Path to ID-trained explainer checkpoint
        output_dir: Output directory for results
        n_eval_samples: Number of evaluation samples
        force_recompute_activations: Force recomputation of activations
        use_wandb: Use wandb for logging
        wandb_project: Wandb project name
        wandb_name: Wandb run name
        baselines: List of baselines to evaluate ['fixed', 'ood_retrained', 'term', 'gae', 'saeboost']
                   If None, evaluates the historical default set: ['fixed', 'ood_retrained', 'term', 'gae']
        ood_retrained_max_epochs: Max epochs for OOD-retrained baseline
        ood_retrained_max_steps: Max training steps for OOD-retrained baseline
        ood_retrained_total_tokens: Total training tokens for OOD-retrained baseline
        ood_retrained_lr: Learning rate for OOD-retrained baseline
        ood_retrained_lambda_sparse: L1 regularization for OOD-retrained baseline
        ood_retrained_objective: Training objective for OOD-retrained baseline (ERM or TERM)
        ood_retrained_term_t: TERM tilt parameter (if objective is TERM)
        ood_retrained_batch_size: Training batch size for OOD-retrained baseline
        ood_retrained_streaming: Use streaming OOD data for retraining
        ood_retrained_context_size: Context size for streaming OOD retraining
        ood_retrained_warm_start: Warm-start from ID explainer weights
        ood_retrained_reinit_b_dec: Reinitialize b_dec when retraining (SAELens only)
        ood_retrained_use_saelens: Use SAELens model for OOD retraining
        term_checkpoint_path: Optional TERM checkpoint override. If omitted,
            the main --checkpoint is evaluated as the TERM explainer.
        gae_r: GAE subspace rank
        adv_ood_token_budget: OOD activation-pool token budget for adversarial short-prompt stability
        adv_bootstrap_repeats: Number of bootstrap repetitions for CI reporting
        adv_bootstrap_ci: Bootstrap confidence level (percent)
    """
    # Set seed
    set_seed(Config.SEED)
    
    # Validate and set baselines
    available_baselines = ['fixed', 'fixed_refit', 'ood_retrained', 'ood_finetuned', 'term', 'gae', 'saeboost', 'faithfulsae']
    default_baselines = ['fixed', 'ood_retrained', 'term', 'gae']
    if baselines is None:
        baselines = default_baselines  # Preserve historical default set
    else:
        # Validate baseline names
        invalid = [b for b in baselines if b not in available_baselines]
        if invalid:
            raise ValueError(f"Invalid baseline(s): {invalid}. Available: {available_baselines}")
        baselines = list(set(baselines))  # Remove duplicates
    
    print(f"Baselines to evaluate: {', '.join(baselines)}")
    
    # Initialize wandb if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Install with: pip install wandb")
            use_wandb = False
        else:
            if wandb_name is None:
                wandb_name = f"{explainer_type}_{model_name}_{ood_set}"
            
            wandb_config = {
                'model': model_name,
                'explainer': explainer_type,
                'ood_set': ood_set,
                'n_eval_samples': n_eval_samples,
                'seed': Config.SEED,
                'baselines': baselines,
            }
            # Add hyperparameters for each baseline
            if 'ood_retrained' in baselines:
                wandb_config.update({
                    'ood_retrained_max_epochs': ood_retrained_max_epochs,
                    'ood_retrained_max_steps': ood_retrained_max_steps,
                    'ood_retrained_total_tokens': ood_retrained_total_tokens,
                    'ood_retrained_lr': ood_retrained_lr,
                    'ood_retrained_lambda_sparse': ood_retrained_lambda_sparse,
                    'ood_retrained_objective': ood_retrained_objective,
                    'ood_retrained_term_t': ood_retrained_term_t,
                    'ood_retrained_batch_size': ood_retrained_batch_size,
                    'ood_retrained_streaming': ood_retrained_streaming,
                    'ood_retrained_context_size': ood_retrained_context_size,
                    'ood_retrained_warm_start': ood_retrained_warm_start,
                    'ood_retrained_reinit_b_dec': ood_retrained_reinit_b_dec,
                    'ood_retrained_use_saelens': ood_retrained_use_saelens,
                    'ood_retrained_checkpoint_path': ood_retrained_checkpoint_path,
                    'ood_retrained_checkpoint_dir': ood_retrained_checkpoint_dir,
                })
            if 'ood_finetuned' in baselines:
                wandb_config.update({
                    'ood_finetuned_total_tokens': ood_finetuned_total_tokens,
                    'ood_finetuned_lr': ood_finetuned_lr,
                    'ood_finetuned_lambda_sparse': ood_finetuned_lambda_sparse,
                    'ood_finetuned_batch_size': ood_finetuned_batch_size,
                    'ood_finetuned_context_size': ood_finetuned_context_size,
                    'ood_finetuned_loss_type': ood_finetuned_loss_type,
                    'ood_finetuned_checkpoint_path': ood_finetuned_checkpoint_path,
                    'ood_finetuned_checkpoint_dir': ood_finetuned_checkpoint_dir,
                })
            if 'term' in baselines:
                wandb_config.update({
                    'term_checkpoint_path': term_checkpoint_path,
                })
            if 'faithfulsae' in baselines:
                wandb_config.update({
                    'faithfulsae_checkpoint_path': faithfulsae_checkpoint_path,
                })
            if 'gae' in baselines:
                wandb_config.update({
                    'gae_r': gae_r,
                    'gae_rank_mode': gae_rank_mode,
                    'gae_rank_energy': gae_rank_energy,
                    'gae_rank_delta_mode': gae_rank_delta_mode,
                    'gae_rank_min': gae_rank_min,
                    'gae_rank_max': gae_rank_max,
                    'gae_recon_lambda': float(gae_recon_lambda),
                })
            wandb_config.update({
                'adv_ood_token_budget': adv_ood_token_budget,
                'adv_bootstrap_repeats': adv_bootstrap_repeats,
                'adv_bootstrap_ci': adv_bootstrap_ci,
            })
            if 'saeboost' in baselines:
                wandb_config.update({
                    'saeboost_residual_checkpoint': saeboost_residual_checkpoint,
                    'saeboost_residual_dir': saeboost_residual_dir,
                    'saeboost_coef': saeboost_coef,
                    'saeboost_train_residual': saeboost_train_residual,
                    'saeboost_residual_max_epochs': saeboost_residual_max_epochs,
                    'saeboost_residual_max_steps': saeboost_residual_max_steps,
                    'saeboost_residual_total_tokens': saeboost_residual_total_tokens,
                    'saeboost_residual_lr': saeboost_residual_lr,
                    'saeboost_residual_lambda_sparse': saeboost_residual_lambda_sparse,
                    'saeboost_residual_objective': saeboost_residual_objective,
                    'saeboost_residual_term_t': saeboost_residual_term_t,
                    'saeboost_residual_batch_size': saeboost_residual_batch_size,
                    'saeboost_residual_context_size': saeboost_residual_context_size,
                    'saeboost_residual_dict_size': saeboost_residual_dict_size,
                    'saeboost_residual_reinit_b_dec': saeboost_residual_reinit_b_dec,
                    'saeboost_residual_sparse_method': saeboost_residual_sparse_method,
                    'saeboost_residual_batch_top_k': saeboost_residual_batch_top_k,
                    'saeboost_residual_batch_top_k_start': saeboost_residual_batch_top_k_start,
                    'saeboost_residual_batch_top_k_warmup_fraction': saeboost_residual_batch_top_k_warmup_fraction,
                    'saeboost_base_infer_sparse_method': saeboost_base_infer_sparse_method,
                    'saeboost_base_infer_batch_top_k': saeboost_base_infer_batch_top_k,
                    'saeboost_base_jumprelu_threshold': saeboost_base_jumprelu_threshold,
                    'saeboost_residual_infer_sparse_method': saeboost_residual_infer_sparse_method,
                    'saeboost_residual_infer_batch_top_k': saeboost_residual_infer_batch_top_k,
                    'saeboost_residual_jumprelu_threshold': saeboost_residual_jumprelu_threshold,
                })
            
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=wandb_config
            )
    
    # Print device info
    print("=" * 80)
    print("Device Information")
    print("=" * 80)
    Config.print_device_info()
    print()
    
    # Load model
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)
    layer = get_layer(model_name)
    d_model = model.cfg.d_model
    vocab_size = model.cfg.d_vocab
    
    print(f"Model config: d_model={d_model}, vocab_size={vocab_size}, layer={layer}")
    
    # Load ID-trained explainer
    print(f"\nLoading ID-trained explainer: {explainer_checkpoint_path}")
    explainer_id = load_explainer(
        explainer_checkpoint_path, explainer_type, d_model, vocab_size
    )
    
    # Load OOD dataset for training/adaptation (train split)
    print(f"\nLoading OOD dataset (train split) for activations: {ood_set}")
    ood_dataset_train, text_field = load_ood_dataset(ood_set, split="train")
    # Buffer streaming/iterable datasets to allow multiple passes (retry fallback)
    if hasattr(ood_dataset_train, "__iter__") and not hasattr(ood_dataset_train, "__getitem__"):
        from datasets import Dataset
        print("Buffering streaming OOD train dataset to memory...")
        ood_dataset_train = Dataset.from_list(list(ood_dataset_train))

    # Load OOD dataset for evaluation (test split)
    print(f"\nLoading OOD dataset (test split) for evaluation: {ood_set}")
    ood_dataset_test, text_field_eval = load_ood_dataset(ood_set, split="test")
    # Buffer streaming eval dataset too
    if hasattr(ood_dataset_test, "__iter__") and not hasattr(ood_dataset_test, "__getitem__"):
        from datasets import Dataset
        print("Buffering streaming OOD test dataset to memory...")
        ood_dataset_test = Dataset.from_list(list(ood_dataset_test))
    
    # Use a shorter min length for short adversarial prompts
    min_len_ood = Config.MIN_LEN
    if min_len_ood_override is not None:
        min_len_ood = int(min_len_ood_override)
    elif ood_set == "halu_eval":
        min_len_ood = min(Config.MIN_LEN, 4)
    elif ood_set == "jailbreakbench":
        min_len_ood = min(Config.MIN_LEN, 8)
    elif ood_set == "jailbreakhub":
        min_len_ood = min(Config.MIN_LEN, 8)
    min_len_eval = Config.MIN_LEN
    if min_len_eval_override is not None:
        min_len_eval = int(min_len_eval_override)
    elif ood_set == "halu_eval":
        min_len_eval = min(Config.MIN_LEN, 4)
    elif ood_set == "jailbreakbench":
        min_len_eval = min(Config.MIN_LEN, 8)
    elif ood_set == "jailbreakhub":
        min_len_eval = min(Config.MIN_LEN, 8)

    # Try to load OOD activations from cache
    ood_data_type = _activation_cache_type(f'ood_adversarial_{ood_set}', explainer_id)
    ood_activations, ood_logits, ood_metadata = None, None, None
    ood_activations_out = None
    ood_cache_metadata = _activation_cache_metadata_for_explainer(
        explainer_id,
        dataset_role="ood",
        task_name="adversarial_ood",
        position=-1,
        ood_set=ood_set,
    )
    
    if not force_recompute_activations:
        ood_activations, ood_logits, ood_metadata = load_activations(
            model_name,
            layer,
            ood_data_type,
            Config.N_COV,
            check_config=True,
            required_metadata=ood_cache_metadata,
        )
        
        # Check if we have enough cached samples
        if ood_activations is not None and len(ood_activations) >= Config.N_COV:
            print(f"Using cached OOD activations ({len(ood_activations)} samples, need {Config.N_COV})")
            ood_activations_full = ood_activations
            # Ensure we have logits for N_COV samples
            if ood_logits is not None and len(ood_logits) >= Config.N_COV:
                ood_logits = ood_logits[:Config.N_COV]
                if getattr(explainer_id, "output_kind", None) == "mlp_out":
                    ood_activations_out = ood_logits
            else:
                print("Warning: Cached logits insufficient, will recompute...")
                ood_activations, ood_logits, ood_metadata = None, None, None
                ood_activations_out = None
    
    if ood_activations is None or len(ood_activations) < Config.N_COV or force_recompute_activations:
        # Need to extract OOD activations
        print(f"Extracting OOD activations (this may take a while)...")
        
        # Sample OOD sequences from train split using token-budget priority.
        if adv_ood_token_budget is not None and int(adv_ood_token_budget) > 0:
            print(
                f"Sampling OOD sequences with token budget={int(adv_ood_token_budget)} "
                f"(max_samples={Config.N_OOD_DOMAIN})..."
            )
            ood_tokenized, collected_tokens = sample_and_tokenize_dataset_by_token_budget(
                ood_dataset_train,
                text_field,
                tokenizer,
                token_budget=int(adv_ood_token_budget),
                max_length=Config.MAX_LEN,
                min_length=min_len_ood,
                seed=Config.SEED,
                max_samples=Config.N_OOD_DOMAIN,
            )
            print(f"Collected {len(ood_tokenized)} sequences / {collected_tokens} tokens")
        else:
            print(f"Sampling {Config.N_OOD_DOMAIN} OOD sequences from train split...")
            ood_tokenized = sample_and_tokenize_dataset(
                ood_dataset_train, text_field, tokenizer, Config.N_OOD_DOMAIN,
                max_length=Config.MAX_LEN, min_length=min_len_ood,
                seed=Config.SEED
            )
        # Retry with a shorter minimum length if adversarial prompts are too short
        if len(ood_tokenized) == 0:
            fallback_min_len = max(4, min_len_ood // 2)
            print(
                f"Warning: 0 OOD samples at MIN_LEN={min_len_ood}. "
                f"Retrying with MIN_LEN={fallback_min_len}."
            )
            if adv_ood_token_budget is not None and int(adv_ood_token_budget) > 0:
                ood_tokenized, _ = sample_and_tokenize_dataset_by_token_budget(
                    ood_dataset_train,
                    text_field,
                    tokenizer,
                    token_budget=int(adv_ood_token_budget),
                    max_length=Config.MAX_LEN,
                    min_length=fallback_min_len,
                    seed=Config.SEED,
                    max_samples=Config.N_OOD_DOMAIN,
                )
            else:
                ood_tokenized = sample_and_tokenize_dataset(
                    ood_dataset_train, text_field, tokenizer, Config.N_OOD_DOMAIN,
                    max_length=Config.MAX_LEN, min_length=fallback_min_len,
                    seed=Config.SEED
                )
        if len(ood_tokenized) == 0:
            # Final fallback: accept any length >= 1
            print("Warning: 0 OOD samples after fallback. Retrying with MIN_LEN=1.")
            if adv_ood_token_budget is not None and int(adv_ood_token_budget) > 0:
                ood_tokenized, _ = sample_and_tokenize_dataset_by_token_budget(
                    ood_dataset_train,
                    text_field,
                    tokenizer,
                    token_budget=int(adv_ood_token_budget),
                    max_length=Config.MAX_LEN,
                    min_length=1,
                    seed=Config.SEED,
                    max_samples=Config.N_OOD_DOMAIN,
                )
            else:
                ood_tokenized = sample_and_tokenize_dataset(
                    ood_dataset_train, text_field, tokenizer, Config.N_OOD_DOMAIN,
                    max_length=Config.MAX_LEN, min_length=1,
                    seed=Config.SEED
                )
        
        if getattr(explainer_id, "output_kind", None) == "mlp_out":
            print("Extracting OOD MLP in/out activations...")
            ood_activations, ood_activations_out = collect_mlp_in_out_activations_batch(
                model,
                ood_tokenized,
                layer,
                position=-1,
                hook_in=getattr(explainer_id, "input_hook_point", "ln2.hook_normalized"),
                hook_out=getattr(explainer_id, "output_hook_point", "hook_mlp_out"),
                device=Config.device,
            )
            ood_logits = ood_activations_out
        else:
            # SAE branch: respect explainer's input_hook_point so the SAE
            # encoder is fed activations from the same hook it was trained on.
            # Re-fix of B3 (docs/sae_failure_v1.md): default was "resid_post",
            # which silently mismatched SAEs trained on ln2.hook_normalized.
            sae_hook = getattr(explainer_id, "input_hook_point", None)
            if sae_hook:
                print(f"Extracting OOD activations (SAE hook='{sae_hook}')...")
                ood_activations = collect_hook_activations_batch(
                    model, ood_tokenized, layer, position=-1,
                    hook_name=sae_hook, device=Config.device,
                )
            else:
                print("Extracting OOD activations (legacy resid_post path)...")
                ood_activations = collect_activations_batch(
                    model, ood_tokenized, layer, position=-1, device=Config.device
                )
            
            # Extract OOD logits
            ood_logits = []
            with torch.no_grad():
                for tokens in ood_tokenized[:Config.N_COV]:
                    tokens_batch = tokens.unsqueeze(0).to(Config.device)
                    logits = extract_logits(model, tokens_batch, position=-1)
                    ood_logits.append(logits.cpu())
            ood_logits = torch.cat(ood_logits, dim=0)
        
        # Save to cache
        save_activations(
            ood_activations[:Config.N_COV],
            ood_logits,
            model_name,
            layer,
            ood_data_type,
            Config.N_COV,
            metadata_extra=ood_cache_metadata,
        )
        
        ood_activations_full = ood_activations
    
    # Use N_COV samples for training/adaptation
    ood_activations = ood_activations_full[:Config.N_COV]
    ood_logits = ood_logits[:Config.N_COV]
    
    # Collect ID activations for GAE (with caching)
    print("Collecting ID activations for GAE...")
    id_activations, id_logits, id_metadata = collect_id_activations(
        model,
        tokenizer,
        Config.N_COV,
        layer,
        use_cache=True,
        force_recompute=force_recompute_activations,
        data_module=data_adv,
    )
    
    # Log activation cache info to wandb
    if use_wandb and WANDB_AVAILABLE:
        cached_n_samples = id_metadata.get('cached_n_samples', len(id_activations))
        wandb.config.update({
            'id_activations_cached_n_samples': cached_n_samples,
            'id_activations_used_n_samples': Config.N_COV,
            'id_activation_cache_used': cached_n_samples >= Config.N_COV
        }, allow_val_change=True)
    
    # Create evaluation prompts from test split
    print("Creating evaluation prompts from test split...")
    if n_eval_samples == -1:
        print("Using all available evaluation samples (n_eval=-1)")
        n_prompts = None
    else:
        n_prompts = n_eval_samples

    eval_prompts, eval_metadata = create_adversarial_prompts(
        ood_dataset_test,
        ood_set,
        tokenizer=tokenizer,
        n_prompts=n_prompts,
        seed=Config.SEED,
        include_choices=True,
        text_field=text_field_eval,
    )
    
    # Tokenize evaluation prompts
    eval_tokens_list = []
    eval_metadata_filtered = []
    # If n_eval_samples == -1, use all prompts; otherwise limit to n_eval_samples
    prompts_to_use = eval_prompts if n_eval_samples == -1 else eval_prompts[:n_eval_samples]
    metadata_to_use = eval_metadata if n_eval_samples == -1 else eval_metadata[:n_eval_samples]
    print(f"Tokenizing {len(prompts_to_use)} prompts (MIN_LEN={min_len_eval}, MAX_LEN={Config.MAX_LEN})...")

    for prompt, meta in zip(prompts_to_use, metadata_to_use):
        tokens = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
        tokens = tokens.squeeze(0)
        if len(tokens) >= min_len_eval:
            eval_tokens_list.append(tokens)
            eval_metadata_filtered.append(meta)
    
    print(f"Created {len(eval_tokens_list)} evaluation sequences (from {len(prompts_to_use)} prompts)")
    if len(eval_tokens_list) == 0:
        print(f"Warning: No valid evaluation sequences! All prompts were filtered out.")
        print(f"  MIN_LEN requirement: {min_len_eval} tokens")
        if len(prompts_to_use) > 0:
            # Debug: check first prompt tokenization
            sample_prompt = prompts_to_use[0]
            sample_tokens = tokenizer.encode(sample_prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            sample_tokens = sample_tokens.squeeze(0)
            print(f"  Sample prompt length: {len(sample_prompt)} chars")
            print(f"  Sample prompt tokens: {len(sample_tokens)} (needs >= {min_len_eval})")
        # Retry with shorter minimum length
        fallback_min_len = max(4, min_len_eval // 2)
        print(f"Retrying eval tokenization with MIN_LEN={fallback_min_len}...")
        eval_tokens_list = []
        eval_metadata_filtered = []
        for prompt, meta in zip(prompts_to_use, metadata_to_use):
            tokens = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            tokens = tokens.squeeze(0)
            if len(tokens) >= fallback_min_len:
                eval_tokens_list.append(tokens)
                eval_metadata_filtered.append(meta)
        print(f"Created {len(eval_tokens_list)} evaluation sequences after fallback")
    if len(eval_tokens_list) == 0:
        print("Retrying eval tokenization with MIN_LEN=1...")
        eval_tokens_list = []
        eval_metadata_filtered = []
        for prompt, meta in zip(prompts_to_use, metadata_to_use):
            tokens = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            tokens = tokens.squeeze(0)
            if len(tokens) >= 1:
                eval_tokens_list.append(tokens)
                eval_metadata_filtered.append(meta)
        print(f"Created {len(eval_tokens_list)} evaluation sequences after MIN_LEN=1")
    if len(eval_tokens_list) == 0:
        raise RuntimeError(
            "No valid evaluation sequences after MIN_LEN fallbacks. "
            "HaluEval prompts may be too short or malformed."
        )
    
    # Prepare target token ids (optional) for adversarial self-consistency
    target_token_ids = None
    if target_mode == "provided":
        target_token_ids = []
        for meta in eval_metadata_filtered[:len(eval_tokens_list)]:
            if isinstance(meta, dict) and meta.get("provided_target_token_id") is not None:
                target_token_ids.append(int(meta["provided_target_token_id"]))
            elif isinstance(meta, dict) and meta.get("incorrect_token_ids"):
                # Legacy compatibility path
                target_token_ids.append(int(meta["incorrect_token_ids"][0]))
            elif isinstance(meta, dict) and meta.get("choice_token_ids"):
                # Legacy compatibility path
                target_token_ids.append(int(meta["choice_token_ids"][0]))
            else:
                target_token_ids.append(None)
        if len(target_token_ids) < len(eval_tokens_list):
            target_token_ids.extend([None] * (len(eval_tokens_list) - len(target_token_ids)))

    # Prepare ID batch for FT computation if requested
    eval_tokens_list_id = None
    if compute_ft:
        print(f"\nPreparing ID batch for Faithfulness Transfer computation...")
        # Load ID dataset and create prompts
        id_dataset, id_text_field = load_id_dataset(model_name)
        id_prompts = create_factual_prompts(
            id_dataset, id_text_field, n_prompts=len(eval_tokens_list), seed=Config.SEED
        )
        # Tokenize ID prompts
        eval_tokens_list_id = []
        for prompt in id_prompts[:len(eval_tokens_list)]:
            tokens = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            tokens = tokens.squeeze(0)
            if len(tokens) >= Config.MIN_LEN:
                eval_tokens_list_id.append(tokens)
        # Match length with OOD batch
        min_len = min(len(eval_tokens_list), len(eval_tokens_list_id))
        eval_tokens_list = eval_tokens_list[:min_len]
        eval_tokens_list_id = eval_tokens_list_id[:min_len]
        eval_metadata_filtered = eval_metadata_filtered[:min_len]
        if target_token_ids is not None:
            target_token_ids = target_token_ids[:min_len]
        print(f"Prepared {len(eval_tokens_list_id)} ID evaluation sequences for FT computation")
    
    # Evaluate baselines
    results = {}

    # Always-compute diagnostics (bounded for runtime)
    DIAG_MAX_SAMPLES = 256
    DIAG_BATCH_SIZE = 16

    def _log_diagnostics_to_wandb(baseline_name: str, diag: dict):
        if not (use_wandb and WANDB_AVAILABLE):
            return
        # Scalar diagnostics
        wandb.log(
            {
                f"diag/{baseline_name}_recon_err_mean": diag.get("recon_err_mean"),
                f"diag/{baseline_name}_recon_err_std": diag.get("recon_err_std"),
                f"diag/{baseline_name}_logit_err0_mean": diag.get("logit_err0_mean"),
                f"diag/{baseline_name}_logit_err0_std": diag.get("logit_err0_std"),
            }
        )
        # Delta-m curve as a table (easy to plot in wandb UI)
        try:
            m_vals = diag.get("m_values", [])
            d_mean = diag.get("delta_curve_mean", [])
            d_std = diag.get("delta_curve_std", [])
            table = wandb.Table(columns=["baseline", "m", "delta_mean", "delta_std"])
            for m, mu, sd in zip(m_vals, d_mean, d_std):
                table.add_data(baseline_name, int(m), float(mu), float(sd))
            wandb.log({f"diag/{baseline_name}_delta_curve": table})
        except Exception:
            # Best-effort: don't crash experiment if wandb plotting fails
            pass
    
    def _print_metric_summary(name: str, res: dict):
        """Print extended metric summary for a baseline."""
        print(
            f"{name} AUC: {res.get('mean_auc', float('nan')):.4f} ± {res.get('std_auc', float('nan')):.4f}"
        )
        if res.get("mean_auc_del") is not None:
            print(
                f"{name} AUC_del: {res.get('mean_auc_del', float('nan')):.4f} ± {res.get('std_auc_del', float('nan')):.4f}"
            )
        if res.get("mean_auc_hidden") is not None:
            print(
                f"{name} Hidden AUC: {res.get('mean_auc_hidden', float('nan')):.4f} ± {res.get('std_auc_hidden', float('nan')):.4f}"
            )
        # m@τ
        if "m_at_tau_abs_mean" in res:
            print(
                f"  m@tau_abs: {res['m_at_tau_abs_mean']:.2f}±{res.get('m_at_tau_abs_std', 0.0):.2f}"
            )
        if "m_at_tau_rel_mean" in res:
            print(
                f"  m@tau_rel: {res['m_at_tau_rel_mean']:.2f}±{res.get('m_at_tau_rel_std', 0.0):.2f}"
            )
        # Sufficiency / Comprehensiveness / CF
        if res.get("suff_mean") is not None:
            print(
                f"  Suff(K): {res['suff_mean']:.3f}±{res.get('suff_std', 0.0):.3f}"
            )
        if res.get("comp_mean") is not None:
            print(
                f"  Comp(K): {res['comp_mean']:.3f}±{res.get('comp_std', 0.0):.3f}"
            )
        if res.get("cf_mean") is not None:
            print(
                f"  CF(K): {res['cf_mean']:.3f}±{res.get('cf_std', 0.0):.3f}"
            )
        # AOPC / Comp / Suff over M (optional)
        if res.get("aopc_mean") is not None:
            print(
                f"  AOPC(M): {res['aopc_mean']:.3f}±{res.get('aopc_std', 0.0):.3f}"
            )
        if res.get("n_aopc_mean") is not None:
            print(
                f"  nAOPC(M): {res['n_aopc_mean']:.3f}±{res.get('n_aopc_std', 0.0):.3f}"
            )
        if res.get("n_suff_mean") is not None:
            print(
                f"  nSuff(M): {res['n_suff_mean']:.3f}±{res.get('n_suff_std', 0.0):.3f}"
            )
        if res.get("n_comp_mean") is not None:
            print(
                f"  nComp(M): {res['n_comp_mean']:.3f}±{res.get('n_comp_std', 0.0):.3f}"
            )
        if res.get("n_aopc_primary_mean") is not None:
            print(
                f"  nAOPC_primary(delta>0): {res['n_aopc_primary_mean']:.3f}"
            )
        if res.get("gap_aopc_mean") is not None:
            print(
                f"  gapAOPC(M): {res['gap_aopc_mean']:.3f}±{res.get('gap_aopc_std', 0.0):.3f}"
            )
        if res.get("comp_m_mean") is not None:
            print(
                f"  Comp(M): {res['comp_m_mean']:.3f}±{res.get('comp_m_std', 0.0):.3f}"
            )
        if res.get("suff_m_mean") is not None:
            print(
                f"  Suff(M): {res['suff_m_mean']:.3f}±{res.get('suff_m_std', 0.0):.3f}"
            )
        if res.get("comp_at_m_mean") is not None:
            print(
                f"  Comp@M*: {res['comp_at_m_mean']:.3f}±{res.get('comp_at_m_std', 0.0):.3f}"
            )
        if res.get("suff_at_m_mean") is not None:
            print(
                f"  Suff@M*: {res['suff_at_m_mean']:.3f}±{res.get('suff_at_m_std', 0.0):.3f}"
            )
        # Spearman ρ
        if res.get("spearman_rho_mean") is not None:
            print(
                f"  Spearman ρ: {res['spearman_rho_mean']:.3f}±{res.get('spearman_rho_std', 0.0):.3f}"
            )
        # Faithfulness Transfer
        if res.get("ft_mean") is not None:
            print(
                f"  FT: {res['ft_mean']:.3f}±{res.get('ft_std', 0.0):.3f}"
            )
        # Geometric metrics (RAER / Hidden-space alignment)
        raer = res.get("raer")
        if isinstance(raer, dict) and raer.get("rare_energy_ratio") is not None:
            print(f"  RAER: {raer['rare_energy_ratio']:.4f}")
        hsa = res.get("hidden_space_alignment")
        if isinstance(hsa, dict) and hsa.get("hidden_space_alignment") is not None:
            print(f"  Hidden-space alignment: {hsa['hidden_space_alignment']:.4f}")
        ewa = res.get("energy_weighted_alignment")
        if isinstance(ewa, dict) and ewa.get("energy_weighted_alignment") is not None:
            print(f"  Energy-weighted alignment: {ewa['energy_weighted_alignment']:.4f}")
        bci = res.get("bootstrap_ci")
        if isinstance(bci, dict) and isinstance(bci.get("auc"), dict):
            auc_ci = bci["auc"]
            print(
                f"  AUC {auc_ci.get('ci_level', 95):.0f}% CI: "
                f"[{auc_ci.get('ci_lower', float('nan')):.4f}, {auc_ci.get('ci_upper', float('nan')):.4f}]"
            )
        behavior_ci = res.get("behavior_bootstrap_ci")
        if isinstance(behavior_ci, dict) and isinstance(behavior_ci.get("auc"), dict):
            b_auc_ci = behavior_ci["auc"]
            print(
                f"  Behavior-level AUC {b_auc_ci.get('ci_level', 95):.0f}% CI: "
                f"[{b_auc_ci.get('ci_lower', float('nan')):.4f}, {b_auc_ci.get('ci_upper', float('nan')):.4f}] "
                f"(n_behaviors={b_auc_ci.get('n_behaviors', 0)})"
            )

    def _log_metrics_to_wandb(baseline_name: str, res: dict):
        """Log extended metrics for a baseline to wandb."""
        if not (use_wandb and WANDB_AVAILABLE):
            return
        log_dict = {
            # Store metrics with baseline-agnostic keys; later baselines overwrite.
            "metrics/auc": res.get("mean_auc"),
            "metrics/auc_std": res.get("std_auc"),
            "metrics/auc_keep": res.get("mean_auc_keep"),
            "metrics/auc_keep_std": res.get("std_auc_keep"),
            "metrics/auc_del": res.get("mean_auc_del"),
            "metrics/auc_del_std": res.get("std_auc_del"),
            "metrics/auc_hidden": res.get("mean_auc_hidden"),
            "metrics/auc_hidden_std": res.get("std_auc_hidden"),
            "metrics/m_at_tau_abs_mean": res.get("m_at_tau_abs_mean"),
            "metrics/m_at_tau_abs_std": res.get("m_at_tau_abs_std"),
            "metrics/m_at_tau_rel_mean": res.get("m_at_tau_rel_mean"),
            "metrics/m_at_tau_rel_std": res.get("m_at_tau_rel_std"),
            "metrics/spearman_rho_mean": res.get("spearman_rho_mean"),
            "metrics/spearman_rho_std": res.get("spearman_rho_std"),
            "metrics/suff_mean": res.get("suff_mean"),
            "metrics/suff_std": res.get("suff_std"),
            "metrics/comp_mean": res.get("comp_mean"),
            "metrics/comp_std": res.get("comp_std"),
            "metrics/cf_mean": res.get("cf_mean"),
            "metrics/cf_std": res.get("cf_std"),
            # AOPC / Comp / Suff over M (optional)
            "metrics/aopc_mean": res.get("aopc_mean"),
            "metrics/aopc_std": res.get("aopc_std"),
            "metrics/aopc_rand_mean": res.get("aopc_rand_mean"),
            "metrics/aopc_rand_std": res.get("aopc_rand_std"),
            "metrics/gap_aopc_mean": res.get("gap_aopc_mean"),
            "metrics/gap_aopc_std": res.get("gap_aopc_std"),
            "metrics/n_aopc_mean": res.get("n_aopc_mean"),
            "metrics/n_aopc_std": res.get("n_aopc_std"),
            "metrics/n_suff_mean": res.get("n_suff_mean"),
            "metrics/n_suff_std": res.get("n_suff_std"),
            "metrics/n_comp_mean": res.get("n_comp_mean"),
            "metrics/n_comp_std": res.get("n_comp_std"),
            "metrics/n_aopc_primary_mean": res.get("n_aopc_primary_mean"),
            "metrics/n_aopc_mean_delta_pos": res.get("n_aopc_mean_delta_pos"),
            "metrics/n_aopc_std_delta_pos": res.get("n_aopc_std_delta_pos"),
            "metrics/n_aopc_mean_delta_nonpos": res.get("n_aopc_mean_delta_nonpos"),
            "metrics/n_aopc_std_delta_nonpos": res.get("n_aopc_std_delta_nonpos"),
            "metrics/n_delta_pos_samples": res.get("n_delta_pos_samples"),
            "metrics/n_delta_nonpos_samples": res.get("n_delta_nonpos_samples"),
            "metrics/gap_n_aopc_mean": res.get("gap_n_aopc_mean"),
            "metrics/gap_n_aopc_std": res.get("gap_n_aopc_std"),
            "metrics/comp_m_mean": res.get("comp_m_mean"),
            "metrics/comp_m_std": res.get("comp_m_std"),
            "metrics/comp_m_primary_mean": res.get("comp_m_primary_mean"),
            "metrics/comp_m_mean_delta_pos": res.get("comp_m_mean_delta_pos"),
            "metrics/comp_m_std_delta_pos": res.get("comp_m_std_delta_pos"),
            "metrics/comp_m_mean_delta_nonpos": res.get("comp_m_mean_delta_nonpos"),
            "metrics/comp_m_std_delta_nonpos": res.get("comp_m_std_delta_nonpos"),
            "metrics/suff_m_mean": res.get("suff_m_mean"),
            "metrics/suff_m_std": res.get("suff_m_std"),
            "metrics/n_comp_m_mean": res.get("n_comp_m_mean"),
            "metrics/n_comp_m_std": res.get("n_comp_m_std"),
            "metrics/n_suff_m_mean": res.get("n_suff_m_mean"),
            "metrics/n_suff_m_std": res.get("n_suff_m_std"),
            "metrics/comp_at_m_mean": res.get("comp_at_m_mean"),
            "metrics/comp_at_m_std": res.get("comp_at_m_std"),
            "metrics/suff_at_m_mean": res.get("suff_at_m_mean"),
            "metrics/suff_at_m_std": res.get("suff_at_m_std"),
            "metrics/delta_max_mean": res.get("delta_max_mean"),
            "metrics/delta_max_std": res.get("delta_max_std"),
            "metrics/delta_max_nonpos_frac": res.get("delta_max_nonpos_frac"),
        }
        # Geometric metrics
        raer = res.get("raer")
        if isinstance(raer, dict):
            log_dict["metrics/raer"] = raer.get("rare_energy_ratio")
        hsa = res.get("hidden_space_alignment")
        if isinstance(hsa, dict):
            log_dict["metrics/hidden_space_alignment"] = hsa.get("hidden_space_alignment")
        ewa = res.get("energy_weighted_alignment")
        if isinstance(ewa, dict):
            log_dict["metrics/energy_weighted_alignment"] = ewa.get("energy_weighted_alignment")
        # FT metrics (only log if computed, i.e., not None)
        if res.get("ft_mean") is not None:
            log_dict["metrics/ft_mean"] = res.get("ft_mean")
            log_dict["metrics/ft_std"] = res.get("ft_std")
        behavior_ci = res.get("behavior_bootstrap_ci")
        if isinstance(behavior_ci, dict) and isinstance(behavior_ci.get("auc"), dict):
            b_auc_ci = behavior_ci["auc"]
            log_dict["metrics/behavior_auc_mean"] = b_auc_ci.get("mean")
            log_dict["metrics/behavior_auc_ci_lower"] = b_auc_ci.get("ci_lower")
            log_dict["metrics/behavior_auc_ci_upper"] = b_auc_ci.get("ci_upper")
            log_dict["metrics/behavior_auc_n_behaviors"] = b_auc_ci.get("n_behaviors")
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
        if log_dict:
            wandb.log(log_dict)

    eval_subsets: Dict[str, List[int]] = {}
    if ood_set == "halu_eval":
        print("Building HaluEval label subsets (hallucinated / non-hallucinated)...")
        eval_subsets = _build_halueval_label_subsets(eval_metadata_filtered)
        hallu_n = len(eval_subsets.get("hallucinated_only", []))
        non_hallu_n = len(eval_subsets.get("non_hallucinated_only", []))
        print(
            f"HaluEval subsets: hallucinated_only={hallu_n}, "
            f"non_hallucinated_only={non_hallu_n}, total={len(eval_tokens_list)}"
        )

    def _subset_list(items: Optional[List], indices: List[int]) -> Optional[List]:
        if items is None:
            return None
        return [items[i] for i in indices if i < len(items)]

    def _evaluate_metadata_subsets(explainer_obj) -> Dict[str, dict]:
        subset_results: Dict[str, dict] = {}
        if not eval_subsets:
            return subset_results
        for subset_name, indices in eval_subsets.items():
            if len(indices) == 0:
                continue
            subset_tokens = _subset_list(eval_tokens_list, indices) or []
            subset_batch_id = _subset_list(eval_tokens_list_id, indices) if compute_ft else None
            subset_targets = _subset_list(target_token_ids, indices) if target_token_ids is not None else None
            sub_res = evaluate_causal_faithfulness(
                model,
                explainer_obj,
                subset_tokens,
                layer,
                explainer_type,
                target_mode=target_mode,
                batch_id=subset_batch_id,
                target_token_ids=subset_targets,
                faith_m_values=faith_m_values,
                faith_random_repeats=faith_random_repeats,
                faith_m_star=faith_m_star,
            )
            _add_faithfulness_bootstrap_ci(
                sub_res,
                n_bootstrap=adv_bootstrap_repeats,
                ci_level=adv_bootstrap_ci,
                seed=Config.SEED,
            )
            subset_results[subset_name] = sub_res
        return subset_results
    
    # (i) Fixed baseline
    if 'fixed' in baselines:
        print("\n" + "="*80)
        print("Evaluating Fixed baseline")
        print("="*80)
        explainer_fixed = explainer_id
        results['fixed'] = evaluate_baseline_fixed(
            explainer_fixed,
            model,
            eval_tokens_list,
            layer,
            explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_faithfulness_bootstrap_ci(
            results['fixed'],
            n_bootstrap=adv_bootstrap_repeats,
            ci_level=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        if ood_set == "jailbreakbench":
            _add_behavior_level_bootstrap_ci(
                results['fixed'],
                eval_metadata_filtered,
                n_bootstrap=adv_bootstrap_repeats,
                ci_level=adv_bootstrap_ci,
                seed=Config.SEED,
            )
        _add_geometric_metrics(
            results['fixed'],
            explainer_fixed,
            id_activations,
            ood_activations,
            explainer_type,
            bootstrap_repeats=adv_bootstrap_repeats,
            bootstrap_ci=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        subset_metrics = _evaluate_metadata_subsets(explainer_fixed)
        if subset_metrics:
            results['fixed']['subsets'] = subset_metrics
        _print_metric_summary("Fixed", results['fixed'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_id,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_id),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['fixed']['diagnostics'] = diag
        print(
            f"[Diagnostics][fixed] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/fixed_auc': results['fixed']['mean_auc']})
            _log_metrics_to_wandb("fixed", results['fixed'])
            _log_diagnostics_to_wandb("fixed", diag)
    
    # (ii) OOD-retrained baseline
    if 'ood_retrained' in baselines:
        print("\n" + "="*80)
        print("Evaluating OOD-retrained baseline")
        print("="*80)
        saelens_dataset_path = None
        saelens_dataset_name = None
        saelens_text_field = text_field
        explainer_ood = evaluate_baseline_ood_retrained(
            model, ood_activations[:Config.N_COV], ood_logits[:Config.N_COV],
            layer, explainer_type, d_model, vocab_size,
            max_epochs=ood_retrained_max_epochs,
            max_steps=ood_retrained_max_steps,
            total_tokens=ood_retrained_total_tokens,
            lr=ood_retrained_lr,
            lambda_sparse=ood_retrained_lambda_sparse,
            objective_type=ood_retrained_objective,
            term_t=ood_retrained_term_t,
            batch_size=ood_retrained_batch_size,
            use_streaming=ood_retrained_streaming,
            ood_dataset=ood_dataset_train,
            ood_text_field=saelens_text_field,
            ood_dataset_path=saelens_dataset_path,
            ood_dataset_name=saelens_dataset_name,
            context_size=ood_retrained_context_size,
            input_hook_point=getattr(explainer_id, "input_hook_point", None),
            output_hook_point=getattr(explainer_id, "output_hook_point", None),
            warm_start=ood_retrained_warm_start,
            reinit_b_dec=ood_retrained_reinit_b_dec,
            warm_start_explainer=explainer_id,
            warm_start_path=explainer_checkpoint_path,
            use_saelens=ood_retrained_use_saelens,
            use_wandb=use_wandb,
            output_kind=getattr(explainer_id, "output_kind", None),
            task_name="adversarial_ood",
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            retrained_checkpoint_path=ood_retrained_checkpoint_path,
            retrained_checkpoint_dir=ood_retrained_checkpoint_dir,
        )
        results['ood_retrained'] = evaluate_causal_faithfulness(
            model,
            explainer_ood,
            eval_tokens_list,
            layer,
            explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_faithfulness_bootstrap_ci(
            results['ood_retrained'],
            n_bootstrap=adv_bootstrap_repeats,
            ci_level=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        if ood_set == "jailbreakbench":
            _add_behavior_level_bootstrap_ci(
                results['ood_retrained'],
                eval_metadata_filtered,
                n_bootstrap=adv_bootstrap_repeats,
                ci_level=adv_bootstrap_ci,
                seed=Config.SEED,
            )
        _add_geometric_metrics(
            results['ood_retrained'],
            explainer_ood,
            id_activations,
            ood_activations,
            explainer_type,
            bootstrap_repeats=adv_bootstrap_repeats,
            bootstrap_ci=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        subset_metrics = _evaluate_metadata_subsets(explainer_ood)
        if subset_metrics:
            results['ood_retrained']['subsets'] = subset_metrics
        _print_metric_summary("OOD-retrained", results['ood_retrained'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_ood,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_ood),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['ood_retrained']['diagnostics'] = diag
        print(
            f"[Diagnostics][ood_retrained] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/ood_retrained_auc': results['ood_retrained']['mean_auc']})
            _log_metrics_to_wandb("ood_retrained", results['ood_retrained'])
            _log_diagnostics_to_wandb("ood_retrained", diag)

    # OOD-Finetuned baseline (Kissane 2024 recipe — warm-start fine-tune)
    if 'ood_finetuned' in baselines:
        print("\n" + "="*80)
        print("Evaluating OOD-Finetuned baseline")
        print("="*80)
        saelens_dataset_path = None
        saelens_dataset_name = None
        saelens_text_field = text_field
        explainer_ft = evaluate_baseline_ood_finetuned(
            model, ood_activations[:Config.N_COV], ood_logits[:Config.N_COV],
            layer, explainer_type, d_model, vocab_size,
            total_tokens=ood_finetuned_total_tokens,
            lr=ood_finetuned_lr,
            lambda_sparse=ood_finetuned_lambda_sparse,
            batch_size=ood_finetuned_batch_size,
            context_size=ood_finetuned_context_size,
            loss_type=ood_finetuned_loss_type,
            ood_dataset=ood_dataset_train,
            ood_text_field=saelens_text_field,
            ood_dataset_path=saelens_dataset_path,
            ood_dataset_name=saelens_dataset_name,
            input_hook_point=getattr(explainer_id, "input_hook_point", None),
            output_hook_point=getattr(explainer_id, "output_hook_point", None),
            warm_start_explainer=explainer_id,
            warm_start_path=explainer_checkpoint_path,
            use_wandb=use_wandb,
            output_kind=getattr(explainer_id, "output_kind", None),
            task_name="adversarial_ood",
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            finetuned_checkpoint_path=ood_finetuned_checkpoint_path,
            finetuned_checkpoint_dir=ood_finetuned_checkpoint_dir,
        )
        results['ood_finetuned'] = evaluate_causal_faithfulness(
            model,
            explainer_ft,
            eval_tokens_list,
            layer,
            explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_faithfulness_bootstrap_ci(
            results['ood_finetuned'],
            n_bootstrap=adv_bootstrap_repeats,
            ci_level=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        if ood_set == "jailbreakbench":
            _add_behavior_level_bootstrap_ci(
                results['ood_finetuned'],
                eval_metadata_filtered,
                n_bootstrap=adv_bootstrap_repeats,
                ci_level=adv_bootstrap_ci,
                seed=Config.SEED,
            )
        _add_geometric_metrics(
            results['ood_finetuned'],
            explainer_ft,
            id_activations,
            ood_activations,
            explainer_type,
            bootstrap_repeats=adv_bootstrap_repeats,
            bootstrap_ci=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        subset_metrics = _evaluate_metadata_subsets(explainer_ft)
        if subset_metrics:
            results['ood_finetuned']['subsets'] = subset_metrics
        _print_metric_summary("OOD-Finetuned", results['ood_finetuned'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_ft,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_ft),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['ood_finetuned']['diagnostics'] = diag
        print(
            f"[Diagnostics][ood_finetuned] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/ood_finetuned_auc': results['ood_finetuned']['mean_auc']})
            _log_metrics_to_wandb("ood_finetuned", results['ood_finetuned'])
            _log_diagnostics_to_wandb("ood_finetuned", diag)

    # (iii) TERM baseline
    if 'term' in baselines:
        print("\n" + "="*80)
        print("Evaluating TERM baseline")
        print("="*80)
        term_source_path = term_checkpoint_path or explainer_checkpoint_path
        explainer_term = evaluate_baseline_term(
            term_source_path,
            explainer_type,
            d_model,
            vocab_size
        )
        results['term'] = evaluate_causal_faithfulness(
            model,
            explainer_term,
            eval_tokens_list,
            layer,
            explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_faithfulness_bootstrap_ci(
            results['term'],
            n_bootstrap=adv_bootstrap_repeats,
            ci_level=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        if ood_set == "jailbreakbench":
            _add_behavior_level_bootstrap_ci(
                results['term'],
                eval_metadata_filtered,
                n_bootstrap=adv_bootstrap_repeats,
                ci_level=adv_bootstrap_ci,
                seed=Config.SEED,
            )
        _add_geometric_metrics(
            results['term'],
            explainer_term,
            id_activations,
            ood_activations,
            explainer_type,
            bootstrap_repeats=adv_bootstrap_repeats,
            bootstrap_ci=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        subset_metrics = _evaluate_metadata_subsets(explainer_term)
        if subset_metrics:
            results['term']['subsets'] = subset_metrics
        _print_metric_summary("TERM", results['term'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_term,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_term),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['term']['diagnostics'] = diag
        print(
            f"[Diagnostics][term] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/term_auc': results['term']['mean_auc']})
            _log_metrics_to_wandb("term", results['term'])
            _log_diagnostics_to_wandb("term", diag)

    # (iv) SAEBoost baseline
    if 'saeboost' in baselines:
        print("\n" + "="*80)
        print("Evaluating SAEBoost baseline")
        print("="*80)
        # Adversarial OOD text_field is a callable; SAEBoost streaming training
        # needs a string key. Flatten dataset to a simple "text" column.
        _saeboost_dataset = ood_dataset_train
        _saeboost_text_field = text_field
        if callable(text_field):
            from datasets import Dataset as _Dataset
            print("Flattening adversarial dataset for SAEBoost streaming training...")
            _texts = [text_field(ex) for ex in ood_dataset_train if text_field(ex) is not None]
            _saeboost_dataset = _Dataset.from_dict({"text": _texts})
            _saeboost_text_field = "text"
            print(f"  Flattened {len(_texts)} examples with 'text' field.")
        explainer_saeboost = evaluate_baseline_saeboost(
            base_explainer=explainer_id,
            model=model,
            layer=layer,
            ood_dataset=_saeboost_dataset,
            ood_text_field=_saeboost_text_field,
            model_type=explainer_type,
            d_model=d_model,
            vocab_size=vocab_size,
            task_name="adversarial_ood",
            model_name=model_name,
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            residual_checkpoint_path=saeboost_residual_checkpoint,
            residual_checkpoint_dir=saeboost_residual_dir,
            resid_coef=saeboost_coef,
            train_residual_if_missing=saeboost_train_residual,
            residual_train_max_epochs=saeboost_residual_max_epochs,
            residual_train_max_steps=saeboost_residual_max_steps,
            residual_train_total_tokens=saeboost_residual_total_tokens,
            residual_train_lr=saeboost_residual_lr,
            residual_train_lambda_sparse=saeboost_residual_lambda_sparse,
            residual_train_objective=saeboost_residual_objective,
            residual_train_term_t=saeboost_residual_term_t,
            residual_train_batch_size=saeboost_residual_batch_size,
            residual_train_context_size=saeboost_residual_context_size,
            residual_train_dict_size=saeboost_residual_dict_size,
            residual_train_reinit_b_dec=saeboost_residual_reinit_b_dec,
            residual_train_sparse_method=saeboost_residual_sparse_method,
            residual_train_batch_top_k=saeboost_residual_batch_top_k,
            residual_train_batch_top_k_start=saeboost_residual_batch_top_k_start,
            residual_train_batch_top_k_warmup_fraction=saeboost_residual_batch_top_k_warmup_fraction,
            base_inference_sparse_method=saeboost_base_infer_sparse_method,
            base_inference_batch_top_k=saeboost_base_infer_batch_top_k,
            base_jumprelu_threshold=saeboost_base_jumprelu_threshold,
            residual_inference_sparse_method=saeboost_residual_infer_sparse_method,
            residual_inference_batch_top_k=saeboost_residual_infer_batch_top_k,
            residual_jumprelu_threshold=saeboost_residual_jumprelu_threshold,
            use_wandb=use_wandb,
        )
        results['saeboost'] = evaluate_causal_faithfulness(
            model,
            explainer_saeboost,
            eval_tokens_list,
            layer,
            explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_faithfulness_bootstrap_ci(
            results['saeboost'],
            n_bootstrap=adv_bootstrap_repeats,
            ci_level=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        if ood_set == "jailbreakbench":
            _add_behavior_level_bootstrap_ci(
                results['saeboost'],
                eval_metadata_filtered,
                n_bootstrap=adv_bootstrap_repeats,
                ci_level=adv_bootstrap_ci,
                seed=Config.SEED,
            )
        _add_geometric_metrics(
            results['saeboost'],
            explainer_saeboost,
            id_activations,
            ood_activations,
            explainer_type,
            bootstrap_repeats=adv_bootstrap_repeats,
            bootstrap_ci=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        subset_metrics = _evaluate_metadata_subsets(explainer_saeboost)
        if subset_metrics:
            results['saeboost']['subsets'] = subset_metrics
        _print_metric_summary("SAEBoost", results['saeboost'])
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_saeboost,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_saeboost),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['saeboost']['diagnostics'] = diag
        print(
            f"[Diagnostics][saeboost] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/saeboost_auc': results['saeboost']['mean_auc']})
            _log_metrics_to_wandb("saeboost", results['saeboost'])
            _log_diagnostics_to_wandb("saeboost", diag)
    

    # (iv-b) FaithfulSAE baseline
    if 'faithfulsae' in baselines:
        faithfulsae_ckpt = faithfulsae_checkpoint_path
        if faithfulsae_ckpt is None:
            print("WARNING: --faithfulsae_checkpoint not provided. Skipping FaithfulSAE baseline.")
        else:
            print("\n" + "="*80)
            print("Evaluating FaithfulSAE baseline")
            print("="*80)
            explainer_faithfulsae = evaluate_baseline_faithfulsae(
                faithfulsae_ckpt,
                explainer_type,
                d_model,
                vocab_size
            )
            results['faithfulsae'] = evaluate_causal_faithfulness(
                model,
                explainer_faithfulsae,
                eval_tokens_list,
                layer,
                explainer_type,
                target_mode=target_mode,
                batch_id=eval_tokens_list_id if compute_ft else None,
                target_token_ids=target_token_ids,
                faith_m_values=faith_m_values,
                faith_random_repeats=faith_random_repeats,
                faith_m_star=faith_m_star,
            )
            _add_faithfulness_bootstrap_ci(
                results['faithfulsae'],
                n_bootstrap=adv_bootstrap_repeats,
                ci_level=adv_bootstrap_ci,
                seed=Config.SEED,
            )
            if ood_set == "jailbreakbench":
                _add_behavior_level_bootstrap_ci(
                    results['faithfulsae'],
                    eval_metadata_filtered,
                    n_bootstrap=adv_bootstrap_repeats,
                    ci_level=adv_bootstrap_ci,
                    seed=Config.SEED,
                )
            _add_geometric_metrics(
                results['faithfulsae'],
                explainer_faithfulsae,
                id_activations,
                ood_activations,
                explainer_type,
                bootstrap_repeats=adv_bootstrap_repeats,
                bootstrap_ci=adv_bootstrap_ci,
                seed=Config.SEED,
            )
            subset_metrics = _evaluate_metadata_subsets(explainer_faithfulsae)
            if subset_metrics:
                results['faithfulsae']['subsets'] = subset_metrics
            _print_metric_summary("FaithfulSAE", results['faithfulsae'])
            # Diagnostics
            diag = compute_ood_diagnostics(
                model=model,
                explainer=explainer_faithfulsae,
                tokens_list=eval_tokens_list,
                layer=layer,
                model_type=explainer_type,
                **_diagnostics_kwargs(explainer_faithfulsae),
                batch_size=DIAG_BATCH_SIZE,
                max_samples=DIAG_MAX_SAMPLES,
            )
            results['faithfulsae']['diagnostics'] = diag
            print(
                f"[Diagnostics][faithfulsae] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
                f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
            )
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({'baseline/faithfulsae_auc': results['faithfulsae']['mean_auc']})
                _log_metrics_to_wandb("faithfulsae", results['faithfulsae'])
                _log_diagnostics_to_wandb("faithfulsae", diag)

    # (v) GAE baseline
    if 'gae' in baselines:
        print("\n" + "="*80)
        print("Evaluating GAE baseline")
        print("="*80)
        _gae_ce_w = _compute_gae_ce_weights(model, eval_tokens_list, layer, explainer_id)
        explainer_gae = evaluate_baseline_gae(
            explainer_id,
            model,
            id_activations,
            ood_activations[:Config.N_COV],
            id_activations_out=id_logits[:Config.N_COV] if id_logits is not None else None,
            ood_activations_out=ood_activations_out[:Config.N_COV] if ood_activations_out is not None else None,
            layer=layer,
            model_type=explainer_type,
            r=gae_r,
            rank_mode=gae_rank_mode,
            rank_energy=gae_rank_energy,
            rank_delta_mode=gae_rank_delta_mode,
            rank_min=gae_rank_min,
            rank_max=gae_rank_max,
            decoder_mode=gae_decoder_mode,
            dict_space=gae_dict_space,
            recon_lambda=gae_recon_lambda,
            ce_weights=_gae_ce_w,
        )
        results['gae'] = evaluate_causal_faithfulness(
            model,
            explainer_gae,
            eval_tokens_list,
            layer,
            explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_faithfulness_bootstrap_ci(
            results['gae'],
            n_bootstrap=adv_bootstrap_repeats,
            ci_level=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        if ood_set == "jailbreakbench":
            _add_behavior_level_bootstrap_ci(
                results['gae'],
                eval_metadata_filtered,
                n_bootstrap=adv_bootstrap_repeats,
                ci_level=adv_bootstrap_ci,
                seed=Config.SEED,
            )
        _add_geometric_metrics(
            results['gae'],
            explainer_gae,
            id_activations,
            ood_activations,
            explainer_type,
            bootstrap_repeats=adv_bootstrap_repeats,
            bootstrap_ci=adv_bootstrap_ci,
            seed=Config.SEED,
        )
        subset_metrics = _evaluate_metadata_subsets(explainer_gae)
        if subset_metrics:
            results['gae']['subsets'] = subset_metrics
        results['gae']['gae_metadata'] = {
            'decoder_mode': getattr(explainer_gae, 'gae_decoder_mode', None),
            'rank_mode': getattr(explainer_gae, 'gae_rank_mode', None),
            'rank_eff': getattr(explainer_gae, 'gae_rank_eff', None),
            'geometry_space': getattr(explainer_gae, 'gae_geometry_space', None),
            'geometry_source': getattr(explainer_gae, 'gae_geometry_source', None),
            'dict_space': getattr(explainer_gae, 'gae_dict_space', None),
            'extra': getattr(explainer_gae, 'gae_extra_metadata', None),
        }
        _print_metric_summary("GAE", results['gae'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_gae,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_gae),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['gae']['diagnostics'] = diag
        print(
            f"[Diagnostics][gae] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/gae_auc': results['gae']['mean_auc']})
            _log_metrics_to_wandb("gae", results['gae'])
            _log_diagnostics_to_wandb("gae", diag)
    
    # Save results
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build results summary with all metrics for evaluated baselines
    results_summary = _build_results_summary(
        task_name='adversarial_ood',
        model_name=model_name,
        explainer_type=explainer_type,
        ood_set=ood_set,
        layer=layer,
        baselines=baselines,
        explainer=explainer_id,
        config_extra={
            'target_mode': target_mode,
            'compute_ft': bool(compute_ft),
            'gae_r': int(gae_r),
            'gae_rank_mode': gae_rank_mode,
            'gae_rank_energy': float(gae_rank_energy),
            'gae_rank_delta_mode': gae_rank_delta_mode,
            'gae_rank_min': int(gae_rank_min),
            'gae_rank_max': int(gae_rank_max) if gae_rank_max is not None else None,
            'gae_decoder_mode': gae_decoder_mode,
            'gae_dict_space': gae_dict_space,
            'gae_recon_lambda': float(gae_recon_lambda),
            'n_eval_samples': int(n_eval_samples),
        },
    )
    
    # Add full metrics for evaluated baselines
    for baseline in baselines:
        if baseline in results:
            results_summary['baselines'][baseline] = results[baseline]
    
    # Log summary to wandb (only for evaluated baselines)
    if use_wandb and WANDB_AVAILABLE:
        wandb_summary = {}
        for baseline in baselines:
            if baseline in results:
                wandb_summary[f'baseline/{baseline}_auc'] = results[baseline]['mean_auc']
        wandb.summary.update(wandb_summary)
        wandb.finish()
    
    # Compute Delta CE if requested
    # Compute Delta CE for all baselines
    _explainer_map = {}
    if 'fixed' in baselines:
        _explainer_map['fixed'] = explainer_id
    if 'term' in baselines and 'explainer_term' in locals():
        _explainer_map['term'] = explainer_term
    if 'saeboost' in baselines and 'explainer_saeboost' in locals():
        _explainer_map['saeboost'] = explainer_saeboost
    if 'gae' in baselines and 'explainer_gae' in locals():
        _explainer_map['gae'] = explainer_gae
    if 'faithfulsae' in baselines and 'explainer_faithfulsae' in locals():
        _explainer_map['faithfulsae'] = explainer_faithfulsae
    if 'ood_retrained' in baselines and 'explainer_ood' in locals():
        _explainer_map['ood_retrained'] = explainer_ood
    if 'ood_finetuned' in baselines and 'explainer_ft' in locals():
        _explainer_map['ood_finetuned'] = explainer_ft
    _compute_delta_ce_for_all_baselines(
        results, _explainer_map, model, eval_tokens_list, layer, explainer_type,
    )
    for bl_name in _explainer_map:
        if bl_name in results and bl_name in results_summary.get('baselines', {}):
            results_summary['baselines'][bl_name] = results[bl_name]

    # Create filename with baseline info
    baseline_str = '_'.join(sorted(baselines))
    results_file = os.path.join(
        output_dir,
        f"adversarial_ood_{model_name}_{explainer_type}_{ood_set}_{baseline_str}.json"
    )

    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return results_summary




def run_domain_ood_experiment(
    model_name,
    explainer_type,
    ood_set,
    explainer_checkpoint_path,
    output_dir=None,
    n_eval_samples=1000,
    force_recompute_activations=False,
    use_wandb=False,
    wandb_project='domain-ood',
    wandb_name=None,
    baselines=None,
    domain_ood_jsonl_dir="domain_ood_prompts",
    domain_ood_use_jsonl=True,
    domain_ood_jsonl_streaming=True,
    domain_ood_force_provided_target=False,
    # OOD-retrained hyperparameters
    ood_retrained_max_epochs=50,
    ood_retrained_max_steps=None,
    ood_retrained_total_tokens=None,
    ood_retrained_lr=1e-3,
    ood_retrained_lambda_sparse=0.01,
    ood_retrained_objective="ERM",
    ood_retrained_term_t=None,
    ood_retrained_batch_size=None,
    ood_retrained_streaming=False,
    ood_retrained_context_size=128,
    ood_retrained_warm_start=True,
    ood_retrained_reinit_b_dec=True,
    ood_retrained_use_saelens=True,
    ood_retrained_checkpoint_path=None,
    ood_retrained_checkpoint_dir=os.environ.get("REPO_DATA", "./data") + "/checkpoints",
    # OOD-finetuned hyperparameters
    ood_finetuned_total_tokens=5_000_000,
    ood_finetuned_lr=None,
    ood_finetuned_lambda_sparse=None,
    ood_finetuned_batch_size=256,
    ood_finetuned_context_size=128,
    ood_finetuned_loss_type="mse",
    ood_finetuned_checkpoint_path=None,
    ood_finetuned_checkpoint_dir=os.environ.get("REPO_DATA", "./data") + "/checkpoints/ood_finetuned",
    # TERM checkpoint
    term_checkpoint_path=None,
    # FaithfulSAE checkpoint
    faithfulsae_checkpoint_path=None,
    # GAE hyperparameters
    gae_r=64,
    gae_rank_mode="fixed",
    gae_rank_energy=0.99,
    gae_rank_delta_mode="ood",
    gae_rank_min=1,
    gae_rank_max=None,
    gae_decoder_mode="theory",
    gae_dict_space="auto",
    gae_recon_lambda=0.0,
    # SAEBoost checkpoint options
    saeboost_residual_checkpoint=None,
    saeboost_residual_dir="checkpoints",
    saeboost_coef=1.0,
    saeboost_train_residual=False,
    saeboost_residual_max_epochs=50,
    saeboost_residual_max_steps=None,
    saeboost_residual_total_tokens=None,
    saeboost_residual_lr=8e-4,
    saeboost_residual_lambda_sparse=5e-6,
    saeboost_residual_objective="ERM",
    saeboost_residual_term_t=None,
    saeboost_residual_batch_size=None,
    saeboost_residual_context_size=128,
    saeboost_residual_dict_size=1024,
    saeboost_residual_reinit_b_dec=True,
    saeboost_residual_sparse_method="l1",
    saeboost_residual_batch_top_k=5,
    saeboost_residual_batch_top_k_start=None,
    saeboost_residual_batch_top_k_warmup_fraction=0.1,
    saeboost_base_infer_sparse_method="auto",
    saeboost_base_infer_batch_top_k=None,
    saeboost_base_jumprelu_threshold=None,
    saeboost_residual_infer_sparse_method="auto",
    saeboost_residual_infer_batch_top_k=None,
    saeboost_residual_jumprelu_threshold=None,
    # Target mode and FT
    target_mode='argmax',
    compute_ft=True,
    faith_m_values=None,
    faith_random_repeats=5,
    faith_m_star=32,
):
    """
    Run Domain-OOD experiment.

    Args:
        model_name: Model identifier ('gpt2', 'pythia-410m', 'pythia-1.4b')
        explainer_type: 'transcoder' or 'sae'
        ood_set: 'patents' or 'edgar' or 'subtitles' or 'standards'
        explainer_checkpoint_path: Path to ID-trained explainer checkpoint
        output_dir: Output directory for results
        n_eval_samples: Number of evaluation samples
        force_recompute_activations: Force recomputation of activations
        use_wandb: Use wandb for logging
        wandb_project: Wandb project name
        wandb_name: Wandb run name
        baselines: List of baselines to evaluate ['fixed', 'ood_retrained', 'term', 'gae', 'saeboost']
                   If None, evaluates the historical default set: ['fixed', 'ood_retrained', 'term', 'gae']
        domain_ood_jsonl_dir: Directory containing per-domain JSONL files (default used)
        domain_ood_use_jsonl: Whether to use JSONL inputs (default: True)
        domain_ood_jsonl_streaming: Use JSONL for streaming retrain (default: True)
        ood_retrained_max_epochs: Max epochs for OOD-retrained baseline
        ood_retrained_max_steps: Max training steps for OOD-retrained baseline
        ood_retrained_total_tokens: Total training tokens for OOD-retrained baseline
        ood_retrained_lr: Learning rate for OOD-retrained baseline
        ood_retrained_lambda_sparse: L1 regularization for OOD-retrained baseline
        ood_retrained_objective: Training objective for OOD-retrained baseline (ERM or TERM)
        ood_retrained_term_t: TERM tilt parameter (if objective is TERM)
        ood_retrained_batch_size: Training batch size for OOD-retrained baseline
        ood_retrained_streaming: Use streaming OOD data for retraining
        ood_retrained_context_size: Context size for streaming OOD retraining
        ood_retrained_warm_start: Warm-start from ID explainer weights
        ood_retrained_reinit_b_dec: Reinitialize b_dec when retraining (SAELens only)
        ood_retrained_use_saelens: Use SAELens model for OOD retraining
        term_checkpoint_path: Optional TERM checkpoint override. If omitted,
            the main --checkpoint is evaluated as the TERM explainer.
        gae_r: GAE subspace rank
    """
    # Set seed
    set_seed(Config.SEED)
    
    # Validate and set baselines
    available_baselines = ['fixed', 'fixed_refit', 'ood_retrained', 'ood_finetuned', 'term', 'gae', 'saeboost', 'faithfulsae']
    default_baselines = ['fixed', 'ood_retrained', 'term', 'gae']
    if baselines is None:
        baselines = default_baselines  # Preserve historical default set
    else:
        # Validate baseline names
        invalid = [b for b in baselines if b not in available_baselines]
        if invalid:
            raise ValueError(f"Invalid baseline(s): {invalid}. Available: {available_baselines}")
        baselines = list(set(baselines))  # Remove duplicates
    
    print(f"Baselines to evaluate: {', '.join(baselines)}")
    
    # Initialize wandb if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Install with: pip install wandb")
            use_wandb = False
        else:
            if wandb_name is None:
                wandb_name = f"{explainer_type}_{model_name}_{ood_set}"
            
            wandb_config = {
                'model': model_name,
                'explainer': explainer_type,
                'ood_set': ood_set,
                'n_eval_samples': n_eval_samples,
                'seed': Config.SEED,
                'baselines': baselines,
            }
            # Add hyperparameters for each baseline
            if 'ood_retrained' in baselines:
                wandb_config.update({
                    'ood_retrained_max_epochs': ood_retrained_max_epochs,
                    'ood_retrained_max_steps': ood_retrained_max_steps,
                    'ood_retrained_total_tokens': ood_retrained_total_tokens,
                    'ood_retrained_lr': ood_retrained_lr,
                    'ood_retrained_lambda_sparse': ood_retrained_lambda_sparse,
                    'ood_retrained_objective': ood_retrained_objective,
                    'ood_retrained_term_t': ood_retrained_term_t,
                    'ood_retrained_batch_size': ood_retrained_batch_size,
                    'ood_retrained_streaming': ood_retrained_streaming,
                    'ood_retrained_context_size': ood_retrained_context_size,
                    'ood_retrained_warm_start': ood_retrained_warm_start,
                    'ood_retrained_reinit_b_dec': ood_retrained_reinit_b_dec,
                    'ood_retrained_use_saelens': ood_retrained_use_saelens,
                    'ood_retrained_checkpoint_path': ood_retrained_checkpoint_path,
                    'ood_retrained_checkpoint_dir': ood_retrained_checkpoint_dir,
                })
            if 'ood_finetuned' in baselines:
                wandb_config.update({
                    'ood_finetuned_total_tokens': ood_finetuned_total_tokens,
                    'ood_finetuned_lr': ood_finetuned_lr,
                    'ood_finetuned_lambda_sparse': ood_finetuned_lambda_sparse,
                    'ood_finetuned_batch_size': ood_finetuned_batch_size,
                    'ood_finetuned_context_size': ood_finetuned_context_size,
                    'ood_finetuned_loss_type': ood_finetuned_loss_type,
                    'ood_finetuned_checkpoint_path': ood_finetuned_checkpoint_path,
                    'ood_finetuned_checkpoint_dir': ood_finetuned_checkpoint_dir,
                })
            if 'term' in baselines:
                wandb_config.update({
                    'term_checkpoint_path': term_checkpoint_path,
                })
            if 'faithfulsae' in baselines:
                wandb_config.update({
                    'faithfulsae_checkpoint_path': faithfulsae_checkpoint_path,
                })
            if 'gae' in baselines:
                wandb_config.update({
                    'gae_r': gae_r,
                    'gae_rank_mode': gae_rank_mode,
                    'gae_rank_energy': gae_rank_energy,
                    'gae_rank_delta_mode': gae_rank_delta_mode,
                    'gae_rank_min': gae_rank_min,
                    'gae_rank_max': gae_rank_max,
                    'gae_recon_lambda': float(gae_recon_lambda),
                })
            if 'saeboost' in baselines:
                wandb_config.update({
                    'saeboost_residual_checkpoint': saeboost_residual_checkpoint,
                    'saeboost_residual_dir': saeboost_residual_dir,
                    'saeboost_coef': saeboost_coef,
                    'saeboost_train_residual': saeboost_train_residual,
                    'saeboost_residual_max_epochs': saeboost_residual_max_epochs,
                    'saeboost_residual_max_steps': saeboost_residual_max_steps,
                    'saeboost_residual_total_tokens': saeboost_residual_total_tokens,
                    'saeboost_residual_lr': saeboost_residual_lr,
                    'saeboost_residual_lambda_sparse': saeboost_residual_lambda_sparse,
                    'saeboost_residual_objective': saeboost_residual_objective,
                    'saeboost_residual_term_t': saeboost_residual_term_t,
                    'saeboost_residual_batch_size': saeboost_residual_batch_size,
                    'saeboost_residual_context_size': saeboost_residual_context_size,
                    'saeboost_residual_dict_size': saeboost_residual_dict_size,
                    'saeboost_residual_reinit_b_dec': saeboost_residual_reinit_b_dec,
                    'saeboost_residual_sparse_method': saeboost_residual_sparse_method,
                    'saeboost_residual_batch_top_k': saeboost_residual_batch_top_k,
                    'saeboost_residual_batch_top_k_start': saeboost_residual_batch_top_k_start,
                    'saeboost_residual_batch_top_k_warmup_fraction': saeboost_residual_batch_top_k_warmup_fraction,
                    'saeboost_base_infer_sparse_method': saeboost_base_infer_sparse_method,
                    'saeboost_base_infer_batch_top_k': saeboost_base_infer_batch_top_k,
                    'saeboost_base_jumprelu_threshold': saeboost_base_jumprelu_threshold,
                    'saeboost_residual_infer_sparse_method': saeboost_residual_infer_sparse_method,
                    'saeboost_residual_infer_batch_top_k': saeboost_residual_infer_batch_top_k,
                    'saeboost_residual_jumprelu_threshold': saeboost_residual_jumprelu_threshold,
                })
            
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=wandb_config
            )
    
    # Print device info
    print("=" * 80)
    print("Device Information")
    print("=" * 80)
    Config.print_device_info()
    print()
    
    # Load model
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)
    layer = get_layer(model_name)
    d_model = model.cfg.d_model
    vocab_size = model.cfg.d_vocab
    
    print(f"Model config: d_model={d_model}, vocab_size={vocab_size}, layer={layer}")
    
    # Load ID-trained explainer
    print(f"\nLoading ID-trained explainer: {explainer_checkpoint_path}")
    explainer_id = load_explainer(
        explainer_checkpoint_path, explainer_type, d_model, vocab_size
    )

    # JSONL-first path (default)
    use_jsonl = bool(domain_ood_use_jsonl)
    jsonl_path = None
    jsonl_records = []
    if use_jsonl:
        jsonl_dir = Path(domain_ood_jsonl_dir)
        jsonl_path = jsonl_dir / f"{ood_set}_{model_name}.jsonl"
        jsonl_records = _load_domain_ood_jsonl_records(jsonl_path)
        if len(jsonl_records) == 0:
            print(f"Warning: JSONL not found or empty at {jsonl_path}; falling back to HF dataset.")
            use_jsonl = False

    ood_dataset_train = None
    ood_dataset_test = None
    text_field = None
    jsonl_token_dataset = None

    if not use_jsonl:
        # Load OOD dataset for training/adaptation (train split)
        print(f"\nLoading OOD dataset (train split) for activations: {ood_set}")
        ood_dataset_train, text_field = data_domain.load_ood_dataset(ood_set, split="train")
        # Load OOD dataset for evaluation (test split)
        print(f"\nLoading OOD dataset (test split) for evaluation: {ood_set}")
        ood_dataset_test, _ = data_domain.load_ood_dataset(ood_set, split="test")
    else:
        if ood_retrained_streaming and domain_ood_jsonl_streaming:
            jsonl_token_dataset = _records_to_token_dataset(jsonl_records)
            if len(jsonl_token_dataset) == 0:
                raise ValueError("JSONL records are empty; cannot stream-train from JSONL.")
            ood_dataset_train = jsonl_token_dataset
            text_field = "tokens"
            print(f"\nUsing JSONL records for streaming retrain ({len(jsonl_token_dataset)} sequences).")
        elif ood_retrained_streaming:
            # JSONL is used for eval, but streaming retrain needs HF dataset
            print(f"\nLoading OOD dataset (train split) for streaming retrain: {ood_set}")
            ood_dataset_train, text_field = data_domain.load_ood_dataset(ood_set, split="train")

    # Ensure ood_dataset_train is available for baselines that need streaming training
    # (SAEBoost residual training, ood_retrained) even when using JSONL for eval
    if ood_dataset_train is None and baselines and ('saeboost' in baselines or 'ood_retrained' in baselines or 'ood_finetuned' in baselines):
        print(f"\nLoading OOD dataset (train split) for baseline training: {ood_set}")
        ood_dataset_train, text_field = data_domain.load_ood_dataset(ood_set, split="train")

    # Try to load OOD activations from cache
    cache_tag = f'ood_{ood_set}_jsonl' if use_jsonl else f'ood_{ood_set}'
    ood_data_type = _activation_cache_type(cache_tag, explainer_id)
    ood_activations, ood_logits, ood_metadata = None, None, None
    ood_activations_out = None
    ood_cache_metadata = _activation_cache_metadata_for_explainer(
        explainer_id,
        dataset_role="ood",
        task_name="domain_ood",
        position=-1,
        ood_set=ood_set,
        use_jsonl=use_jsonl,
    )
    
    if not force_recompute_activations:
        ood_activations, ood_logits, ood_metadata = load_activations(
            model_name,
            layer,
            ood_data_type,
            Config.N_COV,
            check_config=True,
            required_metadata=ood_cache_metadata,
        )
        
        # Check if we have enough cached samples
        if ood_activations is not None and len(ood_activations) >= Config.N_COV:
            print(f"Using cached OOD activations ({len(ood_activations)} samples, need {Config.N_COV})")
            ood_activations_full = ood_activations
            # Ensure we have logits for N_COV samples
            if ood_logits is not None and len(ood_logits) >= Config.N_COV:
                ood_logits = ood_logits[:Config.N_COV]
                if getattr(explainer_id, "output_kind", None) == "mlp_out":
                    ood_activations_out = ood_logits
            else:
                print("Warning: Cached logits insufficient, will recompute...")
                ood_activations, ood_logits, ood_metadata = None, None, None
                ood_activations_out = None
    
    if ood_activations is None or len(ood_activations) < Config.N_COV or force_recompute_activations:
        # Need to extract OOD activations
        print(f"Extracting OOD activations (this may take a while)...")

        if use_jsonl:
            rng = np.random.RandomState(Config.SEED)
            rng.shuffle(jsonl_records)
            n_act = max(Config.N_OOD_DOMAIN, Config.N_COV)
            act_records = jsonl_records[:n_act]
            ood_tokenized, _ = _records_to_tokens(act_records)
            print(f"Using {len(ood_tokenized)} OOD sequences from JSONL for activations.")
        else:
            # Sample OOD sequences from train split
            print(f"Sampling {Config.N_OOD_DOMAIN} OOD sequences from train split...")
            ood_tokenized = data_domain.sample_and_tokenize_dataset(
                ood_dataset_train, text_field, tokenizer, Config.N_OOD_DOMAIN,
                max_length=Config.MAX_LEN, min_length=Config.MIN_LEN,
                seed=Config.SEED
            )
        
        if getattr(explainer_id, "output_kind", None) == "mlp_out":
            print("Extracting OOD MLP in/out activations...")
            ood_activations, ood_activations_out = collect_mlp_in_out_activations_batch(
                model,
                ood_tokenized,
                layer,
                position=-1,
                hook_in=getattr(explainer_id, "input_hook_point", "ln2.hook_normalized"),
                hook_out=getattr(explainer_id, "output_hook_point", "hook_mlp_out"),
                device=Config.device,
            )
            ood_logits = ood_activations_out
        else:
            # SAE branch: respect explainer's input_hook_point so the SAE
            # encoder is fed activations from the same hook it was trained on.
            # Re-fix of B3 (docs/sae_failure_v1.md): default was "resid_post",
            # which silently mismatched SAEs trained on ln2.hook_normalized.
            sae_hook = getattr(explainer_id, "input_hook_point", None)
            if sae_hook:
                print(f"Extracting OOD activations (SAE hook='{sae_hook}')...")
                ood_activations = collect_hook_activations_batch(
                    model, ood_tokenized, layer, position=-1,
                    hook_name=sae_hook, device=Config.device,
                )
            else:
                print("Extracting OOD activations (legacy resid_post path)...")
                ood_activations = collect_activations_batch(
                    model, ood_tokenized, layer, position=-1, device=Config.device
                )
            
            # Extract OOD logits
            ood_logits = []
            with torch.no_grad():
                for tokens in ood_tokenized[:Config.N_COV]:
                    tokens_batch = tokens.unsqueeze(0).to(Config.device)
                    logits = extract_logits(model, tokens_batch, position=-1)
                    ood_logits.append(logits.cpu())
            ood_logits = torch.cat(ood_logits, dim=0)
        
        # Save to cache
        save_activations(
            ood_activations[:Config.N_COV],
            ood_logits,
            model_name,
            layer,
            ood_data_type,
            Config.N_COV,
            metadata_extra=ood_cache_metadata,
        )
        
        ood_activations_full = ood_activations
    
    # Use N_COV samples for training/adaptation
    ood_activations = ood_activations_full[:Config.N_COV]
    ood_logits = ood_logits[:Config.N_COV]
    
    # Collect ID activations for GAE (with caching)
    print("Collecting ID activations for GAE...")
    id_activations, id_logits, id_metadata = collect_id_activations(
        model,
        tokenizer,
        Config.N_COV,
        layer,
        use_cache=True,
        force_recompute=force_recompute_activations,
        data_module=data_domain,
    )
    
    # Log activation cache info to wandb
    if use_wandb and WANDB_AVAILABLE:
        cached_n_samples = id_metadata.get('cached_n_samples', len(id_activations))
        wandb.config.update({
            'id_activations_cached_n_samples': cached_n_samples,
            'id_activations_used_n_samples': Config.N_COV,
            'id_activation_cache_used': cached_n_samples >= Config.N_COV
        }, allow_val_change=True)
    
    # Create evaluation prompts
    print("Creating evaluation prompts...")
    eval_tokens_list = []
    eval_target_token_ids = None
    if use_jsonl:
        rng = np.random.RandomState(Config.SEED + 1)
        rng.shuffle(jsonl_records)
        if n_eval_samples == -1:
            eval_records = jsonl_records
        else:
            eval_records = jsonl_records[:n_eval_samples]
        eval_tokens_list, eval_target_token_ids = _records_to_tokens(eval_records)
        print(f"Loaded {len(eval_tokens_list)} evaluation sequences from JSONL.")
        # Optionally force target_mode to provided for JSONL inputs
        if domain_ood_force_provided_target and target_mode != 'provided':
            print("Note: forcing target_mode='provided' for JSONL inputs.")
            target_mode = 'provided'
    else:
        # Create evaluation prompts from test split
        # If n_eval_samples == -1, use all available samples
        if n_eval_samples == -1:
            print("Using all available evaluation samples (n_eval=-1)")
            # For streaming datasets, we need to iterate to get the count
            if hasattr(ood_dataset_test, '__iter__') and not hasattr(ood_dataset_test, '__len__'):
                # Streaming dataset: collect all prompts
                eval_prompts = data_domain.create_factual_prompts(
                    ood_dataset_test, text_field, n_prompts=None, seed=Config.SEED
                )
            else:
                # Regular dataset: use all samples
                eval_prompts = data_domain.create_factual_prompts(
                    ood_dataset_test, text_field, n_prompts=len(ood_dataset_test), seed=Config.SEED
                )
        else:
            eval_prompts = data_domain.create_factual_prompts(
                ood_dataset_test, text_field, n_prompts=n_eval_samples, seed=Config.SEED
            )

        # Tokenize evaluation prompts
        prompts_to_use = eval_prompts if n_eval_samples == -1 else eval_prompts[:n_eval_samples]
        print(f"Tokenizing {len(prompts_to_use)} prompts (MIN_LEN={Config.MIN_LEN}, MAX_LEN={Config.MAX_LEN})...")
        
        for prompt in prompts_to_use:
            tokens = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            tokens = tokens.squeeze(0)
            if len(tokens) >= Config.MIN_LEN:
                eval_tokens_list.append(tokens)
        
        print(f"Created {len(eval_tokens_list)} evaluation sequences (from {len(prompts_to_use)} prompts)")
    if len(eval_tokens_list) == 0:
        print(f"Warning: No valid evaluation sequences! All prompts were filtered out.")
        print(f"  MIN_LEN requirement: {Config.MIN_LEN} tokens")
        if len(prompts_to_use) > 0:
            # Debug: check first prompt tokenization
            sample_prompt = prompts_to_use[0]
            sample_tokens = tokenizer.encode(sample_prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            sample_tokens = sample_tokens.squeeze(0)
            print(f"  Sample prompt length: {len(sample_prompt)} chars")
            print(f"  Sample prompt tokens: {len(sample_tokens)} (needs >= {Config.MIN_LEN})")
    
    # Prepare ID batch for FT computation if requested
    eval_tokens_list_id = None
    if compute_ft:
        print(f"\nPreparing ID batch for Faithfulness Transfer computation...")
        # Load ID dataset and create prompts
        id_dataset, id_text_field = data_domain.load_id_dataset(model_name)
        id_prompts = data_domain.create_factual_prompts(
            id_dataset, id_text_field, n_prompts=len(eval_tokens_list), seed=Config.SEED
        )
        # Tokenize ID prompts
        eval_tokens_list_id = []
        for prompt in id_prompts[:len(eval_tokens_list)]:
            tokens = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            tokens = tokens.squeeze(0)
            if len(tokens) >= Config.MIN_LEN:
                eval_tokens_list_id.append(tokens)
        # Match length with OOD batch
        min_len = min(len(eval_tokens_list), len(eval_tokens_list_id))
        eval_tokens_list = eval_tokens_list[:min_len]
        eval_tokens_list_id = eval_tokens_list_id[:min_len]
        print(f"Prepared {len(eval_tokens_list_id)} ID evaluation sequences for FT computation")
    
    # Evaluate baselines
    results = {}

    # Always-compute diagnostics (bounded for runtime)
    DIAG_MAX_SAMPLES = 256
    DIAG_BATCH_SIZE = 16

    def _log_diagnostics_to_wandb(baseline_name: str, diag: dict):
        if not (use_wandb and WANDB_AVAILABLE):
            return
        # Scalar diagnostics
        wandb.log(
            {
                f"diag/{baseline_name}_recon_err_mean": diag.get("recon_err_mean"),
                f"diag/{baseline_name}_recon_err_std": diag.get("recon_err_std"),
                f"diag/{baseline_name}_logit_err0_mean": diag.get("logit_err0_mean"),
                f"diag/{baseline_name}_logit_err0_std": diag.get("logit_err0_std"),
            }
        )
        # Delta-m curve as a table (easy to plot in wandb UI)
        try:
            m_vals = diag.get("m_values", [])
            d_mean = diag.get("delta_curve_mean", [])
            d_std = diag.get("delta_curve_std", [])
            table = wandb.Table(columns=["baseline", "m", "delta_mean", "delta_std"])
            for m, mu, sd in zip(m_vals, d_mean, d_std):
                table.add_data(baseline_name, int(m), float(mu), float(sd))
            wandb.log({f"diag/{baseline_name}_delta_curve": table})
        except Exception:
            # Best-effort: don't crash experiment if wandb plotting fails
            pass
    
    def _print_metric_summary(name: str, res: dict):
        """Print extended metric summary for a baseline."""
        print(
            f"{name} AUC: {res.get('mean_auc', float('nan')):.4f} ± {res.get('std_auc', float('nan')):.4f}"
        )
        if res.get("mean_auc_del") is not None:
            print(
                f"{name} AUC_del: {res.get('mean_auc_del', float('nan')):.4f} ± {res.get('std_auc_del', float('nan')):.4f}"
            )
        if res.get("mean_auc_hidden") is not None:
            print(
                f"{name} Hidden AUC: {res.get('mean_auc_hidden', float('nan')):.4f} ± {res.get('std_auc_hidden', float('nan')):.4f}"
            )
        # m@τ
        if "m_at_tau_abs_mean" in res:
            print(
                f"  m@tau_abs: {res['m_at_tau_abs_mean']:.2f}±{res.get('m_at_tau_abs_std', 0.0):.2f}"
            )
        if "m_at_tau_rel_mean" in res:
            print(
                f"  m@tau_rel: {res['m_at_tau_rel_mean']:.2f}±{res.get('m_at_tau_rel_std', 0.0):.2f}"
            )
        # Sufficiency / Comprehensiveness / CF
        if res.get("suff_mean") is not None:
            print(
                f"  Suff(K): {res['suff_mean']:.3f}±{res.get('suff_std', 0.0):.3f}"
            )
        if res.get("comp_mean") is not None:
            print(
                f"  Comp(K): {res['comp_mean']:.3f}±{res.get('comp_std', 0.0):.3f}"
            )
        if res.get("cf_mean") is not None:
            print(
                f"  CF(K): {res['cf_mean']:.3f}±{res.get('cf_std', 0.0):.3f}"
            )
        # AOPC / Comp / Suff over M (optional)
        if res.get("aopc_mean") is not None:
            print(
                f"  AOPC(M): {res['aopc_mean']:.3f}±{res.get('aopc_std', 0.0):.3f}"
            )
        if res.get("n_aopc_mean") is not None:
            print(
                f"  nAOPC(M): {res['n_aopc_mean']:.3f}±{res.get('n_aopc_std', 0.0):.3f}"
            )
        if res.get("n_suff_mean") is not None:
            print(
                f"  nSuff(M): {res['n_suff_mean']:.3f}±{res.get('n_suff_std', 0.0):.3f}"
            )
        if res.get("n_comp_mean") is not None:
            print(
                f"  nComp(M): {res['n_comp_mean']:.3f}±{res.get('n_comp_std', 0.0):.3f}"
            )
        if res.get("n_aopc_primary_mean") is not None:
            print(
                f"  nAOPC_primary(delta>0): {res['n_aopc_primary_mean']:.3f}"
            )
        if res.get("gap_aopc_mean") is not None:
            print(
                f"  gapAOPC(M): {res['gap_aopc_mean']:.3f}±{res.get('gap_aopc_std', 0.0):.3f}"
            )
        if res.get("comp_m_mean") is not None:
            print(
                f"  Comp(M): {res['comp_m_mean']:.3f}±{res.get('comp_m_std', 0.0):.3f}"
            )
        if res.get("suff_m_mean") is not None:
            print(
                f"  Suff(M): {res['suff_m_mean']:.3f}±{res.get('suff_m_std', 0.0):.3f}"
            )
        if res.get("comp_at_m_mean") is not None:
            print(
                f"  Comp@M*: {res['comp_at_m_mean']:.3f}±{res.get('comp_at_m_std', 0.0):.3f}"
            )
        if res.get("suff_at_m_mean") is not None:
            print(
                f"  Suff@M*: {res['suff_at_m_mean']:.3f}±{res.get('suff_at_m_std', 0.0):.3f}"
            )
        # Spearman ρ
        if res.get("spearman_rho_mean") is not None:
            print(
                f"  Spearman ρ: {res['spearman_rho_mean']:.3f}±{res.get('spearman_rho_std', 0.0):.3f}"
            )
        # Faithfulness Transfer
        if res.get("ft_mean") is not None:
            print(
                f"  FT: {res['ft_mean']:.3f}±{res.get('ft_std', 0.0):.3f}"
            )

        # Geometric metrics (RAER / Hidden-space alignment)
        raer = res.get("raer")
        if isinstance(raer, dict) and raer.get("rare_energy_ratio") is not None:
            print(f"  RAER: {raer['rare_energy_ratio']:.4f}")
        hsa = res.get("hidden_space_alignment")
        if isinstance(hsa, dict) and hsa.get("hidden_space_alignment") is not None:
            print(f"  Hidden-space alignment: {hsa['hidden_space_alignment']:.4f}")

    def _log_metrics_to_wandb(baseline_name: str, res: dict):
        """Log extended metrics for a baseline to wandb."""
        if not (use_wandb and WANDB_AVAILABLE):
            return
        log_dict = {
            # Store metrics with baseline-agnostic keys; later baselines overwrite.
            "metrics/auc": res.get("mean_auc"),
            "metrics/auc_std": res.get("std_auc"),
            "metrics/auc_keep": res.get("mean_auc_keep"),
            "metrics/auc_keep_std": res.get("std_auc_keep"),
            "metrics/auc_del": res.get("mean_auc_del"),
            "metrics/auc_del_std": res.get("std_auc_del"),
            "metrics/auc_hidden": res.get("mean_auc_hidden"),
            "metrics/auc_hidden_std": res.get("std_auc_hidden"),
            "metrics/m_at_tau_abs_mean": res.get("m_at_tau_abs_mean"),
            "metrics/m_at_tau_abs_std": res.get("m_at_tau_abs_std"),
            "metrics/m_at_tau_rel_mean": res.get("m_at_tau_rel_mean"),
            "metrics/m_at_tau_rel_std": res.get("m_at_tau_rel_std"),
            "metrics/spearman_rho_mean": res.get("spearman_rho_mean"),
            "metrics/spearman_rho_std": res.get("spearman_rho_std"),
            "metrics/suff_mean": res.get("suff_mean"),
            "metrics/suff_std": res.get("suff_std"),
            "metrics/comp_mean": res.get("comp_mean"),
            "metrics/comp_std": res.get("comp_std"),
            "metrics/cf_mean": res.get("cf_mean"),
            "metrics/cf_std": res.get("cf_std"),
            # AOPC / Comp / Suff over M (optional)
            "metrics/aopc_mean": res.get("aopc_mean"),
            "metrics/aopc_std": res.get("aopc_std"),
            "metrics/aopc_rand_mean": res.get("aopc_rand_mean"),
            "metrics/aopc_rand_std": res.get("aopc_rand_std"),
            "metrics/gap_aopc_mean": res.get("gap_aopc_mean"),
            "metrics/gap_aopc_std": res.get("gap_aopc_std"),
            "metrics/n_aopc_mean": res.get("n_aopc_mean"),
            "metrics/n_aopc_std": res.get("n_aopc_std"),
            "metrics/n_suff_mean": res.get("n_suff_mean"),
            "metrics/n_suff_std": res.get("n_suff_std"),
            "metrics/n_comp_mean": res.get("n_comp_mean"),
            "metrics/n_comp_std": res.get("n_comp_std"),
            "metrics/n_aopc_primary_mean": res.get("n_aopc_primary_mean"),
            "metrics/n_aopc_mean_delta_pos": res.get("n_aopc_mean_delta_pos"),
            "metrics/n_aopc_std_delta_pos": res.get("n_aopc_std_delta_pos"),
            "metrics/n_aopc_mean_delta_nonpos": res.get("n_aopc_mean_delta_nonpos"),
            "metrics/n_aopc_std_delta_nonpos": res.get("n_aopc_std_delta_nonpos"),
            "metrics/n_delta_pos_samples": res.get("n_delta_pos_samples"),
            "metrics/n_delta_nonpos_samples": res.get("n_delta_nonpos_samples"),
            "metrics/gap_n_aopc_mean": res.get("gap_n_aopc_mean"),
            "metrics/gap_n_aopc_std": res.get("gap_n_aopc_std"),
            "metrics/comp_m_mean": res.get("comp_m_mean"),
            "metrics/comp_m_std": res.get("comp_m_std"),
            "metrics/comp_m_primary_mean": res.get("comp_m_primary_mean"),
            "metrics/comp_m_mean_delta_pos": res.get("comp_m_mean_delta_pos"),
            "metrics/comp_m_std_delta_pos": res.get("comp_m_std_delta_pos"),
            "metrics/comp_m_mean_delta_nonpos": res.get("comp_m_mean_delta_nonpos"),
            "metrics/comp_m_std_delta_nonpos": res.get("comp_m_std_delta_nonpos"),
            "metrics/suff_m_mean": res.get("suff_m_mean"),
            "metrics/suff_m_std": res.get("suff_m_std"),
            "metrics/n_comp_m_mean": res.get("n_comp_m_mean"),
            "metrics/n_comp_m_std": res.get("n_comp_m_std"),
            "metrics/n_suff_m_mean": res.get("n_suff_m_mean"),
            "metrics/n_suff_m_std": res.get("n_suff_m_std"),
            "metrics/comp_at_m_mean": res.get("comp_at_m_mean"),
            "metrics/comp_at_m_std": res.get("comp_at_m_std"),
            "metrics/suff_at_m_mean": res.get("suff_at_m_mean"),
            "metrics/suff_at_m_std": res.get("suff_at_m_std"),
            "metrics/delta_max_mean": res.get("delta_max_mean"),
            "metrics/delta_max_std": res.get("delta_max_std"),
            "metrics/delta_max_nonpos_frac": res.get("delta_max_nonpos_frac"),
        }
        # Geometric metrics
        raer = res.get("raer")
        if isinstance(raer, dict):
            log_dict["metrics/raer"] = raer.get("rare_energy_ratio")
        hsa = res.get("hidden_space_alignment")
        if isinstance(hsa, dict):
            log_dict["metrics/hidden_space_alignment"] = hsa.get("hidden_space_alignment")
        # FT metrics (only log if computed, i.e., not None)
        if res.get("ft_mean") is not None:
            log_dict["metrics/ft_mean"] = res.get("ft_mean")
            log_dict["metrics/ft_std"] = res.get("ft_std")
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
        if log_dict:
            wandb.log(log_dict)
    
    # (i) Fixed baseline
    if 'fixed' in baselines:
        print("\n" + "="*80)
        print("Evaluating Fixed baseline")
        print("="*80)
        explainer_fixed = explainer_id
        results['fixed'] = evaluate_baseline_fixed(
            explainer_fixed, model, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=eval_target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_geometric_metrics(results['fixed'], explainer_fixed, id_activations, ood_activations, explainer_type)
        _print_metric_summary("Fixed", results['fixed'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_id,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_id),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['fixed']['diagnostics'] = diag
        print(
            f"[Diagnostics][fixed] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/fixed_auc': results['fixed']['mean_auc']})
            _log_metrics_to_wandb("fixed", results['fixed'])
            _log_diagnostics_to_wandb("fixed", diag)
    
    # (ii) OOD-retrained baseline
    if 'ood_retrained' in baselines:
        print("\n" + "="*80)
        print("Evaluating OOD-retrained baseline")
        print("="*80)
        explainer_ood = evaluate_baseline_ood_retrained(
            model, ood_activations[:Config.N_COV], ood_logits[:Config.N_COV],
            layer, explainer_type, d_model, vocab_size,
            max_epochs=ood_retrained_max_epochs,
            max_steps=ood_retrained_max_steps,
            total_tokens=ood_retrained_total_tokens,
            lr=ood_retrained_lr,
            lambda_sparse=ood_retrained_lambda_sparse,
            objective_type=ood_retrained_objective,
            term_t=ood_retrained_term_t,
            batch_size=ood_retrained_batch_size,
            use_streaming=ood_retrained_streaming,
            ood_dataset=ood_dataset_train,
            ood_text_field=text_field,
            context_size=ood_retrained_context_size,
            input_hook_point=getattr(explainer_id, "input_hook_point", None),
            output_hook_point=getattr(explainer_id, "output_hook_point", None),
            warm_start=ood_retrained_warm_start,
            reinit_b_dec=ood_retrained_reinit_b_dec,
            warm_start_explainer=explainer_id,
            use_saelens=ood_retrained_use_saelens,
            use_wandb=use_wandb,
            output_kind=getattr(explainer_id, "output_kind", None),
            task_name="domain_ood",
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            retrained_checkpoint_path=ood_retrained_checkpoint_path,
            retrained_checkpoint_dir=ood_retrained_checkpoint_dir,
        )
        results['ood_retrained'] = evaluate_causal_faithfulness(
            model, explainer_ood, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=eval_target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_geometric_metrics(results['ood_retrained'], explainer_ood, id_activations, ood_activations, explainer_type)
        _print_metric_summary("OOD-retrained", results['ood_retrained'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_ood,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_ood),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['ood_retrained']['diagnostics'] = diag
        print(
            f"[Diagnostics][ood_retrained] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/ood_retrained_auc': results['ood_retrained']['mean_auc']})
            _log_metrics_to_wandb("ood_retrained", results['ood_retrained'])
            _log_diagnostics_to_wandb("ood_retrained", diag)

    # OOD-Finetuned baseline (Kissane 2024 recipe — warm-start fine-tune)
    if 'ood_finetuned' in baselines:
        print("\n" + "="*80)
        print("Evaluating OOD-Finetuned baseline")
        print("="*80)
        explainer_ft = evaluate_baseline_ood_finetuned(
            model, ood_activations[:Config.N_COV], ood_logits[:Config.N_COV],
            layer, explainer_type, d_model, vocab_size,
            total_tokens=ood_finetuned_total_tokens,
            lr=ood_finetuned_lr,
            lambda_sparse=ood_finetuned_lambda_sparse,
            batch_size=ood_finetuned_batch_size,
            context_size=ood_finetuned_context_size,
            loss_type=ood_finetuned_loss_type,
            ood_dataset=ood_dataset_train,
            ood_text_field=text_field,
            ood_dataset_path=None,
            ood_dataset_name=None,
            input_hook_point=getattr(explainer_id, "input_hook_point", None),
            output_hook_point=getattr(explainer_id, "output_hook_point", None),
            warm_start_explainer=explainer_id,
            warm_start_path=explainer_checkpoint_path,
            use_wandb=use_wandb,
            output_kind=getattr(explainer_id, "output_kind", None),
            task_name="domain_ood",
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            finetuned_checkpoint_path=ood_finetuned_checkpoint_path,
            finetuned_checkpoint_dir=ood_finetuned_checkpoint_dir,
        )
        results['ood_finetuned'] = evaluate_causal_faithfulness(
            model, explainer_ft, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=eval_target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_geometric_metrics(results['ood_finetuned'], explainer_ft, id_activations, ood_activations, explainer_type)
        _print_metric_summary("OOD-Finetuned", results['ood_finetuned'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_ft,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_ft),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['ood_finetuned']['diagnostics'] = diag
        print(
            f"[Diagnostics][ood_finetuned] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/ood_finetuned_auc': results['ood_finetuned']['mean_auc']})
            _log_metrics_to_wandb("ood_finetuned", results['ood_finetuned'])
            _log_diagnostics_to_wandb("ood_finetuned", diag)

    # (iii) TERM baseline
    if 'term' in baselines:
        print("\n" + "="*80)
        print("Evaluating TERM baseline")
        print("="*80)
        term_source_path = term_checkpoint_path or explainer_checkpoint_path
        explainer_term = evaluate_baseline_term(
            term_source_path,
            explainer_type,
            d_model,
            vocab_size
        )
        results['term'] = evaluate_causal_faithfulness(
            model, explainer_term, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=eval_target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_geometric_metrics(results['term'], explainer_term, id_activations, ood_activations, explainer_type)
        _print_metric_summary("TERM", results['term'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_term,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_term),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['term']['diagnostics'] = diag
        print(
            f"[Diagnostics][term] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/term_auc': results['term']['mean_auc']})
            _log_metrics_to_wandb("term", results['term'])
            _log_diagnostics_to_wandb("term", diag)

    # (iv) SAEBoost baseline
    if 'saeboost' in baselines:
        print("\n" + "="*80)
        print("Evaluating SAEBoost baseline")
        print("="*80)
        explainer_saeboost = evaluate_baseline_saeboost(
            base_explainer=explainer_id,
            model=model,
            layer=layer,
            ood_dataset=ood_dataset_train,
            ood_text_field=text_field,
            model_type=explainer_type,
            d_model=d_model,
            vocab_size=vocab_size,
            task_name="domain_ood",
            model_name=model_name,
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            residual_checkpoint_path=saeboost_residual_checkpoint,
            residual_checkpoint_dir=saeboost_residual_dir,
            resid_coef=saeboost_coef,
            train_residual_if_missing=saeboost_train_residual,
            residual_train_max_epochs=saeboost_residual_max_epochs,
            residual_train_max_steps=saeboost_residual_max_steps,
            residual_train_total_tokens=saeboost_residual_total_tokens,
            residual_train_lr=saeboost_residual_lr,
            residual_train_lambda_sparse=saeboost_residual_lambda_sparse,
            residual_train_objective=saeboost_residual_objective,
            residual_train_term_t=saeboost_residual_term_t,
            residual_train_batch_size=saeboost_residual_batch_size,
            residual_train_context_size=saeboost_residual_context_size,
            residual_train_dict_size=saeboost_residual_dict_size,
            residual_train_reinit_b_dec=saeboost_residual_reinit_b_dec,
            residual_train_sparse_method=saeboost_residual_sparse_method,
            residual_train_batch_top_k=saeboost_residual_batch_top_k,
            residual_train_batch_top_k_start=saeboost_residual_batch_top_k_start,
            residual_train_batch_top_k_warmup_fraction=saeboost_residual_batch_top_k_warmup_fraction,
            base_inference_sparse_method=saeboost_base_infer_sparse_method,
            base_inference_batch_top_k=saeboost_base_infer_batch_top_k,
            base_jumprelu_threshold=saeboost_base_jumprelu_threshold,
            residual_inference_sparse_method=saeboost_residual_infer_sparse_method,
            residual_inference_batch_top_k=saeboost_residual_infer_batch_top_k,
            residual_jumprelu_threshold=saeboost_residual_jumprelu_threshold,
            use_wandb=use_wandb,
        )
        results['saeboost'] = evaluate_causal_faithfulness(
            model, explainer_saeboost, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=eval_target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_geometric_metrics(results['saeboost'], explainer_saeboost, id_activations, ood_activations, explainer_type)
        _print_metric_summary("SAEBoost", results['saeboost'])
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_saeboost,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_saeboost),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['saeboost']['diagnostics'] = diag
        print(
            f"[Diagnostics][saeboost] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/saeboost_auc': results['saeboost']['mean_auc']})
            _log_metrics_to_wandb("saeboost", results['saeboost'])
            _log_diagnostics_to_wandb("saeboost", diag)
    

    # (iv-b) FaithfulSAE baseline
    if 'faithfulsae' in baselines:
        faithfulsae_ckpt = faithfulsae_checkpoint_path
        if faithfulsae_ckpt is None:
            print("WARNING: --faithfulsae_checkpoint not provided. Skipping FaithfulSAE baseline.")
        else:
            print("\n" + "="*80)
            print("Evaluating FaithfulSAE baseline")
            print("="*80)
            explainer_faithfulsae = evaluate_baseline_faithfulsae(
                faithfulsae_ckpt,
                explainer_type,
                d_model,
                vocab_size
            )
            results['faithfulsae'] = evaluate_causal_faithfulness(
                model, explainer_faithfulsae, eval_tokens_list, layer, explainer_type,
                target_mode=target_mode,
                batch_id=eval_tokens_list_id if compute_ft else None,
                target_token_ids=eval_target_token_ids,
                faith_m_values=faith_m_values,
                faith_random_repeats=faith_random_repeats,
                faith_m_star=faith_m_star,
            )
            _add_geometric_metrics(results['faithfulsae'], explainer_faithfulsae, id_activations, ood_activations, explainer_type)
            _print_metric_summary("FaithfulSAE", results['faithfulsae'])
            # Diagnostics
            diag = compute_ood_diagnostics(
                model=model,
                explainer=explainer_faithfulsae,
                tokens_list=eval_tokens_list,
                layer=layer,
                model_type=explainer_type,
                **_diagnostics_kwargs(explainer_faithfulsae),
                batch_size=DIAG_BATCH_SIZE,
                max_samples=DIAG_MAX_SAMPLES,
            )
            results['faithfulsae']['diagnostics'] = diag
            print(
                f"[Diagnostics][faithfulsae] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
                f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
            )
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({'baseline/faithfulsae_auc': results['faithfulsae']['mean_auc']})
                _log_metrics_to_wandb("faithfulsae", results['faithfulsae'])
                _log_diagnostics_to_wandb("faithfulsae", diag)

    # (v) GAE baseline
    if 'gae' in baselines:
        print("\n" + "="*80)
        print("Evaluating GAE baseline")
        print("="*80)
        _gae_ce_w = _compute_gae_ce_weights(model, eval_tokens_list, layer, explainer_id)
        explainer_gae = evaluate_baseline_gae(
            explainer_id,
            model,
            id_activations,
            ood_activations[:Config.N_COV],
            id_activations_out=id_logits[:Config.N_COV] if id_logits is not None else None,
            ood_activations_out=ood_activations_out[:Config.N_COV] if ood_activations_out is not None else None,
            layer=layer,
            model_type=explainer_type,
            r=gae_r,
            rank_mode=gae_rank_mode,
            rank_energy=gae_rank_energy,
            rank_delta_mode=gae_rank_delta_mode,
            rank_min=gae_rank_min,
            rank_max=gae_rank_max,
            decoder_mode=gae_decoder_mode,
            dict_space=gae_dict_space,
            recon_lambda=gae_recon_lambda,
            ce_weights=_gae_ce_w,
        )
        results['gae'] = evaluate_causal_faithfulness(
            model, explainer_gae, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode,
            batch_id=eval_tokens_list_id if compute_ft else None,
            target_token_ids=eval_target_token_ids,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
        )
        _add_geometric_metrics(results['gae'], explainer_gae, id_activations, ood_activations, explainer_type)
        results['gae']['gae_metadata'] = {
            'decoder_mode': getattr(explainer_gae, 'gae_decoder_mode', None),
            'rank_mode': getattr(explainer_gae, 'gae_rank_mode', None),
            'rank_eff': getattr(explainer_gae, 'gae_rank_eff', None),
            'geometry_space': getattr(explainer_gae, 'gae_geometry_space', None),
            'geometry_source': getattr(explainer_gae, 'gae_geometry_source', None),
            'dict_space': getattr(explainer_gae, 'gae_dict_space', None),
        }
        _print_metric_summary("GAE", results['gae'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_gae,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_gae),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['gae']['diagnostics'] = diag
        print(
            f"[Diagnostics][gae] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/gae_auc': results['gae']['mean_auc']})
            _log_metrics_to_wandb("gae", results['gae'])
            _log_diagnostics_to_wandb("gae", diag)
    
    # Save results
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build results summary with all metrics for evaluated baselines
    results_summary = _build_results_summary(
        task_name='domain_ood',
        model_name=model_name,
        explainer_type=explainer_type,
        ood_set=ood_set,
        layer=layer,
        baselines=baselines,
        explainer=explainer_id,
        config_extra={
            'target_mode': target_mode,
            'compute_ft': bool(compute_ft),
            'gae_r': int(gae_r),
            'gae_rank_mode': gae_rank_mode,
            'gae_rank_energy': float(gae_rank_energy),
            'gae_rank_delta_mode': gae_rank_delta_mode,
            'gae_rank_min': int(gae_rank_min),
            'gae_rank_max': int(gae_rank_max) if gae_rank_max is not None else None,
            'gae_decoder_mode': gae_decoder_mode,
            'gae_dict_space': gae_dict_space,
            'gae_recon_lambda': float(gae_recon_lambda),
            'n_eval_samples': int(n_eval_samples),
            'domain_ood_use_jsonl': bool(domain_ood_use_jsonl),
            'domain_ood_jsonl_streaming': bool(domain_ood_jsonl_streaming),
        },
    )
    
    # Add full metrics for evaluated baselines
    for baseline in baselines:
        if baseline in results:
            results_summary['baselines'][baseline] = results[baseline]
    
    # Log summary to wandb (only for evaluated baselines)
    if use_wandb and WANDB_AVAILABLE:
        wandb_summary = {}
        for baseline in baselines:
            if baseline in results:
                wandb_summary[f'baseline/{baseline}_auc'] = results[baseline]['mean_auc']
        wandb.summary.update(wandb_summary)
        wandb.finish()
    
    # Compute Delta CE if requested
    # Compute Delta CE for all baselines
    _explainer_map = {}
    if 'fixed' in baselines:
        _explainer_map['fixed'] = explainer_id
    if 'term' in baselines and 'explainer_term' in locals():
        _explainer_map['term'] = explainer_term
    if 'saeboost' in baselines and 'explainer_saeboost' in locals():
        _explainer_map['saeboost'] = explainer_saeboost
    if 'gae' in baselines and 'explainer_gae' in locals():
        _explainer_map['gae'] = explainer_gae
    if 'faithfulsae' in baselines and 'explainer_faithfulsae' in locals():
        _explainer_map['faithfulsae'] = explainer_faithfulsae
    if 'ood_retrained' in baselines and 'explainer_ood' in locals():
        _explainer_map['ood_retrained'] = explainer_ood
    if 'ood_finetuned' in baselines and 'explainer_ft' in locals():
        _explainer_map['ood_finetuned'] = explainer_ft
    _compute_delta_ce_for_all_baselines(
        results, _explainer_map, model, eval_tokens_list, layer, explainer_type,
    )
    for bl_name in _explainer_map:
        if bl_name in results and bl_name in results_summary.get('baselines', {}):
            results_summary['baselines'][bl_name] = results[bl_name]

    # Create filename with baseline info
    baseline_str = '_'.join(sorted(baselines))
    results_file = os.path.join(
        output_dir,
        f"domain_ood_{model_name}_{explainer_type}_{ood_set}_{baseline_str}.json"
    )

    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return results_summary


def run_timeshift_ood_experiment(
    model_name,
    explainer_type,
    ood_set,
    explainer_checkpoint_path,
    output_dir=None,
    n_eval_samples=1000,
    force_recompute_activations=False,
    use_wandb=False,
    wandb_project='timeshift-ood',
    wandb_name=None,
    baselines=None,
    # OOD-retrained hyperparameters
    ood_retrained_max_epochs=50,
    ood_retrained_max_steps=None,
    ood_retrained_total_tokens=None,
    ood_retrained_lr=1e-3,
    ood_retrained_lambda_sparse=0.01,
    ood_retrained_objective="ERM",
    ood_retrained_term_t=None,
    ood_retrained_batch_size=None,
    ood_retrained_streaming=False,
    ood_retrained_context_size=128,
    ood_retrained_warm_start=True,
    ood_retrained_reinit_b_dec=True,
    ood_retrained_use_saelens=True,
    ood_retrained_checkpoint_path=None,
    ood_retrained_checkpoint_dir=os.environ.get("REPO_DATA", "./data") + "/checkpoints",
    # OOD-finetuned hyperparameters
    ood_finetuned_total_tokens=5_000_000,
    ood_finetuned_lr=None,
    ood_finetuned_lambda_sparse=None,
    ood_finetuned_batch_size=256,
    ood_finetuned_context_size=128,
    ood_finetuned_loss_type="mse",
    ood_finetuned_checkpoint_path=None,
    ood_finetuned_checkpoint_dir=os.environ.get("REPO_DATA", "./data") + "/checkpoints/ood_finetuned",
    # TERM checkpoint
    term_checkpoint_path=None,
    # FaithfulSAE checkpoint
    faithfulsae_checkpoint_path=None,
    # GAE hyperparameters
    gae_r=64,
    gae_rank_mode="fixed",
    gae_rank_energy=0.99,
    gae_rank_delta_mode="ood",
    gae_rank_min=1,
    gae_rank_max=None,
    gae_decoder_mode="theory",
    gae_dict_space="auto",
    gae_recon_lambda=0.0,
    # SAEBoost checkpoint options
    saeboost_residual_checkpoint=None,
    saeboost_residual_dir="checkpoints",
    saeboost_coef=1.0,
    saeboost_train_residual=False,
    saeboost_residual_max_epochs=50,
    saeboost_residual_max_steps=None,
    saeboost_residual_total_tokens=None,
    saeboost_residual_lr=8e-4,
    saeboost_residual_lambda_sparse=5e-6,
    saeboost_residual_objective="ERM",
    saeboost_residual_term_t=None,
    saeboost_residual_batch_size=None,
    saeboost_residual_context_size=128,
    saeboost_residual_dict_size=1024,
    saeboost_residual_reinit_b_dec=True,
    saeboost_residual_sparse_method="l1",
    saeboost_residual_batch_top_k=5,
    saeboost_residual_batch_top_k_start=None,
    saeboost_residual_batch_top_k_warmup_fraction=0.1,
    saeboost_base_infer_sparse_method="auto",
    saeboost_base_infer_batch_top_k=None,
    saeboost_base_jumprelu_threshold=None,
    saeboost_residual_infer_sparse_method="auto",
    saeboost_residual_infer_batch_top_k=None,
    saeboost_residual_jumprelu_threshold=None,
    # Target mode and FT
    target_mode='argmax',
    compute_ft=True,
    faith_m_values=None,
    faith_random_repeats=5,
    faith_m_star=32,
    faith_empty_mode="zero_resid",
    faith_rank_mode="causal_topz",
    faith_rank_top_f=64,
    faith_hook_site_mode="auto",
    timeshift_span_mode="prefix",
):
    """
    Run Time-shift OOD experiment.

    Args:
        model_name: Model identifier ('gpt2', 'pythia-410m', 'pythia-1.4b')
        explainer_type: 'transcoder' or 'sae'
        ood_set: 'fineweb' or 'dolma_web'
        explainer_checkpoint_path: Path to ID-trained explainer checkpoint
        output_dir: Output directory for results
        n_eval_samples: Number of evaluation samples
        force_recompute_activations: Force recomputation of activations
        use_wandb: Use wandb for logging
        wandb_project: Wandb project name
        wandb_name: Wandb run name
        baselines: List of baselines to evaluate ['fixed', 'ood_retrained', 'term', 'gae', 'saeboost']
                   If None, evaluates the historical default set: ['fixed', 'ood_retrained', 'term', 'gae']
        ood_retrained_max_epochs: Max epochs for OOD-retrained baseline
        ood_retrained_max_steps: Max training steps for OOD-retrained baseline
        ood_retrained_total_tokens: Total training tokens for OOD-retrained baseline
        ood_retrained_lr: Learning rate for OOD-retrained baseline
        ood_retrained_lambda_sparse: L1 regularization for OOD-retrained baseline
        ood_retrained_objective: Training objective for OOD-retrained baseline (ERM or TERM)
        ood_retrained_term_t: TERM tilt parameter (if objective is TERM)
        ood_retrained_batch_size: Training batch size for OOD-retrained baseline
        ood_retrained_streaming: Use streaming OOD data for retraining
        ood_retrained_context_size: Context size for streaming OOD retraining
        ood_retrained_warm_start: Warm-start from ID explainer weights
        ood_retrained_reinit_b_dec: Reinitialize b_dec when retraining (SAELens only)
        ood_retrained_use_saelens: Use SAELens model for OOD retraining
        term_checkpoint_path: Optional TERM checkpoint override. If omitted,
            the main --checkpoint is evaluated as the TERM explainer.
        gae_r: GAE subspace rank
    """
    # Set seed
    set_seed(Config.SEED)

    if timeshift_span_mode not in ("prefix", "middle"):
        raise ValueError("timeshift_span_mode must be one of {'prefix', 'middle'}.")
    if faith_rank_mode not in ("activation", "causal_topz"):
        raise ValueError("faith_rank_mode must be one of {'activation', 'causal_topz'}.")
    if faith_hook_site_mode not in FAITH_HOOK_SITE_MODES:
        raise ValueError(f"faith_hook_site_mode must be one of {FAITH_HOOK_SITE_MODES}.")
    if faith_rank_top_f <= 0:
        raise ValueError("faith_rank_top_f must be positive.")
    
    # Validate and set baselines
    available_baselines = ['fixed', 'fixed_refit', 'ood_retrained', 'ood_finetuned', 'term', 'gae', 'saeboost', 'faithfulsae']
    default_baselines = ['fixed', 'ood_retrained', 'term', 'gae']
    if baselines is None:
        baselines = default_baselines  # Preserve historical default set
    else:
        # Validate baseline names
        invalid = [b for b in baselines if b not in available_baselines]
        if invalid:
            raise ValueError(f"Invalid baseline(s): {invalid}. Available: {available_baselines}")
        baselines = list(set(baselines))  # Remove duplicates

    # Timeshift path does not provide dataset-derived target token IDs.
    if target_mode == 'provided':
        raise ValueError(
            "Time-shift OOD does not support target_mode='provided'. "
            "Use target_mode='argmax' (default) or 'gold'."
        )
    
    print(f"Baselines to evaluate: {', '.join(baselines)}")
    
    # Initialize wandb if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not installed. Install with: pip install wandb")
            use_wandb = False
        else:
            if wandb_name is None:
                wandb_name = f"{explainer_type}_{model_name}_{ood_set}"
            
            wandb_config = {
                'model': model_name,
                'explainer': explainer_type,
                'ood_set': ood_set,
                'n_eval_samples': n_eval_samples,
                'seed': Config.SEED,
                'baselines': baselines,
                'timeshift_span_mode': timeshift_span_mode,
                'faith_empty_mode': faith_empty_mode,
                'faith_rank_mode': faith_rank_mode,
                'faith_rank_top_f': faith_rank_top_f,
                'faith_hook_site_mode': faith_hook_site_mode,
            }
            # Add hyperparameters for each baseline
            if 'ood_retrained' in baselines:
                wandb_config.update({
                    'ood_retrained_max_epochs': ood_retrained_max_epochs,
                    'ood_retrained_max_steps': ood_retrained_max_steps,
                    'ood_retrained_total_tokens': ood_retrained_total_tokens,
                    'ood_retrained_lr': ood_retrained_lr,
                    'ood_retrained_lambda_sparse': ood_retrained_lambda_sparse,
                    'ood_retrained_objective': ood_retrained_objective,
                    'ood_retrained_term_t': ood_retrained_term_t,
                    'ood_retrained_batch_size': ood_retrained_batch_size,
                    'ood_retrained_streaming': ood_retrained_streaming,
                    'ood_retrained_context_size': ood_retrained_context_size,
                    'ood_retrained_warm_start': ood_retrained_warm_start,
                    'ood_retrained_reinit_b_dec': ood_retrained_reinit_b_dec,
                    'ood_retrained_use_saelens': ood_retrained_use_saelens,
                    'ood_retrained_checkpoint_path': ood_retrained_checkpoint_path,
                    'ood_retrained_checkpoint_dir': ood_retrained_checkpoint_dir,
                })
            if 'ood_finetuned' in baselines:
                wandb_config.update({
                    'ood_finetuned_total_tokens': ood_finetuned_total_tokens,
                    'ood_finetuned_lr': ood_finetuned_lr,
                    'ood_finetuned_lambda_sparse': ood_finetuned_lambda_sparse,
                    'ood_finetuned_batch_size': ood_finetuned_batch_size,
                    'ood_finetuned_context_size': ood_finetuned_context_size,
                    'ood_finetuned_loss_type': ood_finetuned_loss_type,
                    'ood_finetuned_checkpoint_path': ood_finetuned_checkpoint_path,
                    'ood_finetuned_checkpoint_dir': ood_finetuned_checkpoint_dir,
                })
            if 'term' in baselines:
                wandb_config.update({
                    'term_checkpoint_path': term_checkpoint_path,
                })
            if 'faithfulsae' in baselines:
                wandb_config.update({
                    'faithfulsae_checkpoint_path': faithfulsae_checkpoint_path,
                })
            if 'gae' in baselines:
                wandb_config.update({
                    'gae_r': gae_r,
                    'gae_rank_mode': gae_rank_mode,
                    'gae_rank_energy': gae_rank_energy,
                    'gae_rank_delta_mode': gae_rank_delta_mode,
                    'gae_rank_min': gae_rank_min,
                    'gae_rank_max': gae_rank_max,
                    'gae_recon_lambda': float(gae_recon_lambda),
                })
            if 'saeboost' in baselines:
                wandb_config.update({
                    'saeboost_residual_checkpoint': saeboost_residual_checkpoint,
                    'saeboost_residual_dir': saeboost_residual_dir,
                    'saeboost_coef': saeboost_coef,
                    'saeboost_train_residual': saeboost_train_residual,
                    'saeboost_residual_max_epochs': saeboost_residual_max_epochs,
                    'saeboost_residual_max_steps': saeboost_residual_max_steps,
                    'saeboost_residual_total_tokens': saeboost_residual_total_tokens,
                    'saeboost_residual_lr': saeboost_residual_lr,
                    'saeboost_residual_lambda_sparse': saeboost_residual_lambda_sparse,
                    'saeboost_residual_objective': saeboost_residual_objective,
                    'saeboost_residual_term_t': saeboost_residual_term_t,
                    'saeboost_residual_batch_size': saeboost_residual_batch_size,
                    'saeboost_residual_context_size': saeboost_residual_context_size,
                    'saeboost_residual_dict_size': saeboost_residual_dict_size,
                    'saeboost_residual_reinit_b_dec': saeboost_residual_reinit_b_dec,
                    'saeboost_residual_sparse_method': saeboost_residual_sparse_method,
                    'saeboost_residual_batch_top_k': saeboost_residual_batch_top_k,
                    'saeboost_residual_batch_top_k_start': saeboost_residual_batch_top_k_start,
                    'saeboost_residual_batch_top_k_warmup_fraction': saeboost_residual_batch_top_k_warmup_fraction,
                    'saeboost_base_infer_sparse_method': saeboost_base_infer_sparse_method,
                    'saeboost_base_infer_batch_top_k': saeboost_base_infer_batch_top_k,
                    'saeboost_base_jumprelu_threshold': saeboost_base_jumprelu_threshold,
                    'saeboost_residual_infer_sparse_method': saeboost_residual_infer_sparse_method,
                    'saeboost_residual_infer_batch_top_k': saeboost_residual_infer_batch_top_k,
                    'saeboost_residual_jumprelu_threshold': saeboost_residual_jumprelu_threshold,
                })
            
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=wandb_config
            )
    
    # Print device info
    print("=" * 80)
    print("Device Information")
    print("=" * 80)
    Config.print_device_info()
    print()
    
    # Load model
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name)
    layer = get_layer(model_name)
    d_model = model.cfg.d_model
    vocab_size = model.cfg.d_vocab
    
    print(f"Model config: d_model={d_model}, vocab_size={vocab_size}, layer={layer}")
    
    # Load ID-trained explainer
    print(f"\nLoading ID-trained explainer: {explainer_checkpoint_path}")
    explainer_id = load_explainer(
        explainer_checkpoint_path, explainer_type, d_model, vocab_size
    )
    
    # Load OOD dataset for training/adaptation (train split)
    # Rate-limit tolerant: if HF API fails but activation cache exists, defer to cache-only path.
    print(f"\nLoading OOD dataset (train split) for activations: {ood_set}")
    try:
        ood_dataset_train, text_field = data_timeshift.load_ood_dataset(ood_set, split="train")
    except Exception as _e_train:
        err_s = str(_e_train)
        if "429" in err_s or "rate limit" in err_s.lower() or "Too Many Requests" in err_s:
            print(f"Warning: HF rate limit on train split load — will rely on activation cache (text unused).")
            ood_dataset_train, text_field = None, None
        else:
            raise

    # Load OOD dataset for evaluation (test split)
    print(f"\nLoading OOD dataset (test split) for evaluation: {ood_set}")
    try:
        ood_dataset_test, _ = data_timeshift.load_ood_dataset(ood_set, split="test")
    except Exception as _e_test:
        err_s = str(_e_test)
        if "429" in err_s or "rate limit" in err_s.lower() or "Too Many Requests" in err_s:
            print(f"Warning: HF rate limit on test split load — will rely on activation cache / ID fallback for eval prompts.")
            ood_dataset_test = None
        else:
            raise
    
    # Try to load OOD activations from cache
    ood_data_type = _activation_cache_type(f'ood_timeshift_{ood_set}', explainer_id)
    ood_activations, ood_logits, ood_metadata = None, None, None
    ood_activations_out = None
    ood_cache_metadata = _activation_cache_metadata_for_explainer(
        explainer_id,
        dataset_role="ood",
        task_name="timeshift_ood",
        position=-1,
        ood_set=ood_set,
        timeshift_span_mode=timeshift_span_mode,
    )
    
    if not force_recompute_activations:
        ood_activations, ood_logits, ood_metadata = load_activations(
            model_name,
            layer,
            ood_data_type,
            Config.N_COV,
            check_config=True,
            required_metadata=ood_cache_metadata,
        )
        
        # Check if we have enough cached samples
        if ood_activations is not None and len(ood_activations) >= Config.N_COV:
            print(f"Using cached OOD activations ({len(ood_activations)} samples, need {Config.N_COV})")
            ood_activations_full = ood_activations
            # Ensure we have logits for N_COV samples
            if ood_logits is not None and len(ood_logits) >= Config.N_COV:
                ood_logits = ood_logits[:Config.N_COV]
                if getattr(explainer_id, "output_kind", None) == "mlp_out":
                    ood_activations_out = ood_logits
            else:
                print("Warning: Cached logits insufficient, will recompute...")
                ood_activations, ood_logits, ood_metadata = None, None, None
                ood_activations_out = None
    
    if ood_activations is None or len(ood_activations) < Config.N_COV or force_recompute_activations:
        # Need to extract OOD activations
        print(f"Extracting OOD activations (this may take a while)...")
        
        # Sample OOD sequences from train split
        print(f"Sampling {Config.N_OOD_DOMAIN} OOD sequences from train split...")
        span_mode = timeshift_span_mode
        print(f"Using timeshift span_mode='{span_mode}' for all models.")
        ood_tokenized = data_timeshift.sample_and_tokenize_dataset(
            ood_dataset_train, text_field, tokenizer, Config.N_OOD_DOMAIN,
            max_length=Config.MAX_LEN, min_length=Config.MIN_LEN,
            seed=Config.SEED, span_mode=span_mode
        )
        if len(ood_tokenized) == 0:
            fallback_min_len = max(8, Config.MIN_LEN // 2)
            print(
                f"Warning: 0 OOD samples at MIN_LEN={Config.MIN_LEN}. "
                f"Retrying with MIN_LEN={fallback_min_len}."
            )
            ood_tokenized = data_timeshift.sample_and_tokenize_dataset(
                ood_dataset_train, text_field, tokenizer, Config.N_OOD_DOMAIN,
                max_length=Config.MAX_LEN, min_length=fallback_min_len,
                seed=Config.SEED, span_mode=span_mode
            )
        if len(ood_tokenized) == 0:
            raise RuntimeError(
                "No OOD samples available for Time-shift activation extraction. "
                "Check OOD dataset contents, text field mapping, and length constraints."
            )
        
        if getattr(explainer_id, "output_kind", None) == "mlp_out":
            print("Extracting OOD MLP in/out activations...")
            ood_activations, ood_activations_out = collect_mlp_in_out_activations_batch(
                model,
                ood_tokenized,
                layer,
                position=-1,
                hook_in=getattr(explainer_id, "input_hook_point", "ln2.hook_normalized"),
                hook_out=getattr(explainer_id, "output_hook_point", "hook_mlp_out"),
                device=Config.device,
            )
            ood_logits = ood_activations_out
        else:
            # SAE branch: respect explainer's input_hook_point so the SAE
            # encoder is fed activations from the same hook it was trained on.
            # Re-fix of B3 (docs/sae_failure_v1.md): default was "resid_post",
            # which silently mismatched SAEs trained on ln2.hook_normalized.
            sae_hook = getattr(explainer_id, "input_hook_point", None)
            if sae_hook:
                print(f"Extracting OOD activations (SAE hook='{sae_hook}')...")
                ood_activations = collect_hook_activations_batch(
                    model, ood_tokenized, layer, position=-1,
                    hook_name=sae_hook, device=Config.device,
                )
            else:
                print("Extracting OOD activations (legacy resid_post path)...")
                ood_activations = collect_activations_batch(
                    model, ood_tokenized, layer, position=-1, device=Config.device
                )
            
            # Extract OOD logits
            ood_logits = []
            with torch.no_grad():
                for tokens in ood_tokenized[:Config.N_COV]:
                    tokens_batch = tokens.unsqueeze(0).to(Config.device)
                    logits = extract_logits(model, tokens_batch, position=-1)
                    ood_logits.append(logits.cpu())
            ood_logits = torch.cat(ood_logits, dim=0)
        
        # Save to cache
        save_activations(
            ood_activations[:Config.N_COV],
            ood_logits,
            model_name,
            layer,
            ood_data_type,
            Config.N_COV,
            metadata_extra=ood_cache_metadata,
        )
        
        ood_activations_full = ood_activations
    
    # Use N_COV samples for training/adaptation
    ood_activations = ood_activations_full[:Config.N_COV]
    ood_logits = ood_logits[:Config.N_COV]
    
    # Collect ID activations for GAE (with caching)
    print("Collecting ID activations for GAE...")
    id_activations, id_logits, id_metadata = collect_id_activations(
        model,
        tokenizer,
        Config.N_COV,
        layer,
        use_cache=True,
        force_recompute=force_recompute_activations,
        data_module=data_timeshift,
    )
    
    # Log activation cache info to wandb
    if use_wandb and WANDB_AVAILABLE:
        cached_n_samples = id_metadata.get('cached_n_samples', len(id_activations))
        wandb.config.update({
            'id_activations_cached_n_samples': cached_n_samples,
            'id_activations_used_n_samples': Config.N_COV,
            'id_activation_cache_used': cached_n_samples >= Config.N_COV
        }, allow_val_change=True)
    
    # Create evaluation prompts from test split
    print("Creating evaluation prompts from test split...")

    def _create_eval_prompts_from_dataset(ds):
        if n_eval_samples == -1:
            print("Using all available evaluation samples (n_eval=-1)")
            # For streaming datasets, we need to iterate to get the count
            if hasattr(ds, '__iter__') and not hasattr(ds, '__len__'):
                return data_timeshift.create_factual_prompts(
                    ds, text_field, n_prompts=None, seed=Config.SEED
                )
            return data_timeshift.create_factual_prompts(
                ds, text_field, n_prompts=len(ds), seed=Config.SEED
            )
        return data_timeshift.create_factual_prompts(
            ds, text_field, n_prompts=n_eval_samples, seed=Config.SEED
        )

    try:
        eval_prompts = _create_eval_prompts_from_dataset(ood_dataset_test)
    except Exception as e:
        err_msg = str(e)
        is_truncated_gzip = (
            isinstance(e, EOFError)
            or "Compressed file ended before the end-of-stream marker" in err_msg
        )
        if ood_set == "dolma_web" and is_truncated_gzip:
            print(
                "Warning: detected truncated Dolma gzip while reading test split. "
                "Retrying with DOLMA_VALIDATE_GZIP=1 (test split only)."
            )
            prev_validate = os.environ.get("DOLMA_VALIDATE_GZIP")
            os.environ["DOLMA_VALIDATE_GZIP"] = "1"
            try:
                ood_dataset_test, _ = data_timeshift.load_ood_dataset(ood_set, split="test")
                eval_prompts = _create_eval_prompts_from_dataset(ood_dataset_test)
            finally:
                if prev_validate is None:
                    os.environ.pop("DOLMA_VALIDATE_GZIP", None)
                else:
                    os.environ["DOLMA_VALIDATE_GZIP"] = prev_validate
        else:
            raise
    
    # Tokenize evaluation prompts
    eval_tokens_list = []
    # If n_eval_samples == -1, use all prompts; otherwise limit to n_eval_samples
    prompts_to_use = eval_prompts if n_eval_samples == -1 else eval_prompts[:n_eval_samples]
    print(f"Tokenizing {len(prompts_to_use)} prompts (MIN_LEN={Config.MIN_LEN}, MAX_LEN={Config.MAX_LEN})...")
    
    for prompt in prompts_to_use:
        tokens = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
        tokens = tokens.squeeze(0)
        if len(tokens) >= Config.MIN_LEN:
            eval_tokens_list.append(tokens)
    
    print(f"Created {len(eval_tokens_list)} evaluation sequences (from {len(prompts_to_use)} prompts)")
    if len(eval_tokens_list) == 0:
        print(f"Warning: No valid evaluation sequences! All prompts were filtered out.")
        print(f"  MIN_LEN requirement: {Config.MIN_LEN} tokens")
        if len(prompts_to_use) > 0:
            # Debug: check first prompt tokenization
            sample_prompt = prompts_to_use[0]
            sample_tokens = tokenizer.encode(sample_prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            sample_tokens = sample_tokens.squeeze(0)
            print(f"  Sample prompt length: {len(sample_prompt)} chars")
            print(f"  Sample prompt tokens: {len(sample_tokens)} (needs >= {Config.MIN_LEN})")
    
    # Prepare ID batch for FT computation if requested
    eval_tokens_list_id = None
    if compute_ft:
        print(f"\nPreparing ID batch for Faithfulness Transfer computation...")
        # Load ID dataset and create prompts
        id_dataset, id_text_field = data_timeshift.load_id_dataset(model_name)
        id_prompts = data_timeshift.create_factual_prompts(
            id_dataset, id_text_field, n_prompts=len(eval_tokens_list), seed=Config.SEED
        )
        # Tokenize ID prompts
        eval_tokens_list_id = []
        for prompt in id_prompts[:len(eval_tokens_list)]:
            tokens = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=Config.MAX_LEN)
            tokens = tokens.squeeze(0)
            if len(tokens) >= Config.MIN_LEN:
                eval_tokens_list_id.append(tokens)
        # Match length with OOD batch
        min_len = min(len(eval_tokens_list), len(eval_tokens_list_id))
        eval_tokens_list = eval_tokens_list[:min_len]
        eval_tokens_list_id = eval_tokens_list_id[:min_len]
        print(f"Prepared {len(eval_tokens_list_id)} ID evaluation sequences for FT computation")
    
    # Evaluate baselines
    results = {}

    # Always-compute diagnostics (bounded for runtime)
    DIAG_MAX_SAMPLES = 256
    DIAG_BATCH_SIZE = 16

    def _log_diagnostics_to_wandb(baseline_name: str, diag: dict):
        if not (use_wandb and WANDB_AVAILABLE):
            return
        # Scalar diagnostics
        wandb.log(
            {
                f"diag/{baseline_name}_recon_err_mean": diag.get("recon_err_mean"),
                f"diag/{baseline_name}_recon_err_std": diag.get("recon_err_std"),
                f"diag/{baseline_name}_logit_err0_mean": diag.get("logit_err0_mean"),
                f"diag/{baseline_name}_logit_err0_std": diag.get("logit_err0_std"),
            }
        )
        # Delta-m curve as a table (easy to plot in wandb UI)
        try:
            m_vals = diag.get("m_values", [])
            d_mean = diag.get("delta_curve_mean", [])
            d_std = diag.get("delta_curve_std", [])
            table = wandb.Table(columns=["baseline", "m", "delta_mean", "delta_std"])
            for m, mu, sd in zip(m_vals, d_mean, d_std):
                table.add_data(baseline_name, int(m), float(mu), float(sd))
            wandb.log({f"diag/{baseline_name}_delta_curve": table})
        except Exception:
            # Best-effort: don't crash experiment if wandb plotting fails
            pass
    
    def _print_metric_summary(name: str, res: dict):
        """Print extended metric summary for a baseline."""
        print(
            f"{name} AUC: {res.get('mean_auc', float('nan')):.4f} ± {res.get('std_auc', float('nan')):.4f}"
        )
        if res.get("mean_auc_del") is not None:
            print(
                f"{name} AUC_del: {res.get('mean_auc_del', float('nan')):.4f} ± {res.get('std_auc_del', float('nan')):.4f}"
            )
        if res.get("mean_auc_hidden") is not None:
            print(
                f"{name} Hidden AUC: {res.get('mean_auc_hidden', float('nan')):.4f} ± {res.get('std_auc_hidden', float('nan')):.4f}"
            )
        # m@τ
        if "m_at_tau_abs_mean" in res:
            print(
                f"  m@tau_abs: {res['m_at_tau_abs_mean']:.2f}±{res.get('m_at_tau_abs_std', 0.0):.2f}"
            )
        if "m_at_tau_rel_mean" in res:
            print(
                f"  m@tau_rel: {res['m_at_tau_rel_mean']:.2f}±{res.get('m_at_tau_rel_std', 0.0):.2f}"
            )
        # Sufficiency / Comprehensiveness / CF
        if res.get("suff_mean") is not None:
            print(
                f"  Suff(K): {res['suff_mean']:.3f}±{res.get('suff_std', 0.0):.3f}"
            )
        if res.get("comp_mean") is not None:
            print(
                f"  Comp(K): {res['comp_mean']:.3f}±{res.get('comp_std', 0.0):.3f}"
            )
        if res.get("cf_mean") is not None:
            print(
                f"  CF(K): {res['cf_mean']:.3f}±{res.get('cf_std', 0.0):.3f}"
            )
        # AOPC / Comp / Suff over M (optional)
        if res.get("aopc_mean") is not None:
            print(
                f"  AOPC(M): {res['aopc_mean']:.3f}±{res.get('aopc_std', 0.0):.3f}"
            )
        if res.get("n_aopc_mean") is not None:
            print(
                f"  nAOPC(M): {res['n_aopc_mean']:.3f}±{res.get('n_aopc_std', 0.0):.3f}"
            )
        if res.get("n_suff_mean") is not None:
            print(
                f"  nSuff(M): {res['n_suff_mean']:.3f}±{res.get('n_suff_std', 0.0):.3f}"
            )
        if res.get("n_comp_mean") is not None:
            print(
                f"  nComp(M): {res['n_comp_mean']:.3f}±{res.get('n_comp_std', 0.0):.3f}"
            )
        if res.get("n_aopc_primary_mean") is not None:
            print(
                f"  nAOPC_primary(delta>0): {res['n_aopc_primary_mean']:.3f}"
            )
        if res.get("gap_aopc_mean") is not None:
            print(
                f"  gapAOPC(M): {res['gap_aopc_mean']:.3f}±{res.get('gap_aopc_std', 0.0):.3f}"
            )
        if res.get("comp_m_mean") is not None:
            print(
                f"  Comp(M): {res['comp_m_mean']:.3f}±{res.get('comp_m_std', 0.0):.3f}"
            )
        if res.get("suff_m_mean") is not None:
            print(
                f"  Suff(M): {res['suff_m_mean']:.3f}±{res.get('suff_m_std', 0.0):.3f}"
            )
        if res.get("comp_at_m_mean") is not None:
            print(
                f"  Comp@M*: {res['comp_at_m_mean']:.3f}±{res.get('comp_at_m_std', 0.0):.3f}"
            )
        if res.get("suff_at_m_mean") is not None:
            print(
                f"  Suff@M*: {res['suff_at_m_mean']:.3f}±{res.get('suff_at_m_std', 0.0):.3f}"
            )
        # Spearman ρ
        if res.get("spearman_rho_mean") is not None:
            print(
                f"  Spearman ρ: {res['spearman_rho_mean']:.3f}±{res.get('spearman_rho_std', 0.0):.3f}"
            )
        # Faithfulness Transfer
        if res.get("ft_mean") is not None:
            print(
                f"  FT: {res['ft_mean']:.3f}±{res.get('ft_std', 0.0):.3f}"
            )

        # Geometric metrics (RAER / Hidden-space alignment)
        raer = res.get("raer")
        if isinstance(raer, dict) and raer.get("rare_energy_ratio") is not None:
            print(f"  RAER: {raer['rare_energy_ratio']:.4f}")
        hsa = res.get("hidden_space_alignment")
        if isinstance(hsa, dict) and hsa.get("hidden_space_alignment") is not None:
            print(f"  Hidden-space alignment: {hsa['hidden_space_alignment']:.4f}")

    def _log_metrics_to_wandb(baseline_name: str, res: dict):
        """Log extended metrics for a baseline to wandb."""
        if not (use_wandb and WANDB_AVAILABLE):
            return
        log_dict = {
            # Store metrics with baseline-agnostic keys; later baselines overwrite.
            "metrics/auc": res.get("mean_auc"),
            "metrics/auc_std": res.get("std_auc"),
            "metrics/auc_keep": res.get("mean_auc_keep"),
            "metrics/auc_keep_std": res.get("std_auc_keep"),
            "metrics/auc_del": res.get("mean_auc_del"),
            "metrics/auc_del_std": res.get("std_auc_del"),
            "metrics/auc_hidden": res.get("mean_auc_hidden"),
            "metrics/auc_hidden_std": res.get("std_auc_hidden"),
            "metrics/m_at_tau_abs_mean": res.get("m_at_tau_abs_mean"),
            "metrics/m_at_tau_abs_std": res.get("m_at_tau_abs_std"),
            "metrics/m_at_tau_rel_mean": res.get("m_at_tau_rel_mean"),
            "metrics/m_at_tau_rel_std": res.get("m_at_tau_rel_std"),
            "metrics/spearman_rho_mean": res.get("spearman_rho_mean"),
            "metrics/spearman_rho_std": res.get("spearman_rho_std"),
            "metrics/suff_mean": res.get("suff_mean"),
            "metrics/suff_std": res.get("suff_std"),
            "metrics/comp_mean": res.get("comp_mean"),
            "metrics/comp_std": res.get("comp_std"),
            "metrics/cf_mean": res.get("cf_mean"),
            "metrics/cf_std": res.get("cf_std"),
            # AOPC / Comp / Suff over M (optional)
            "metrics/aopc_mean": res.get("aopc_mean"),
            "metrics/aopc_std": res.get("aopc_std"),
            "metrics/aopc_rand_mean": res.get("aopc_rand_mean"),
            "metrics/aopc_rand_std": res.get("aopc_rand_std"),
            "metrics/gap_aopc_mean": res.get("gap_aopc_mean"),
            "metrics/gap_aopc_std": res.get("gap_aopc_std"),
            "metrics/n_aopc_mean": res.get("n_aopc_mean"),
            "metrics/n_aopc_std": res.get("n_aopc_std"),
            "metrics/n_suff_mean": res.get("n_suff_mean"),
            "metrics/n_suff_std": res.get("n_suff_std"),
            "metrics/n_comp_mean": res.get("n_comp_mean"),
            "metrics/n_comp_std": res.get("n_comp_std"),
            "metrics/n_aopc_primary_mean": res.get("n_aopc_primary_mean"),
            "metrics/n_aopc_mean_delta_pos": res.get("n_aopc_mean_delta_pos"),
            "metrics/n_aopc_std_delta_pos": res.get("n_aopc_std_delta_pos"),
            "metrics/n_aopc_mean_delta_nonpos": res.get("n_aopc_mean_delta_nonpos"),
            "metrics/n_aopc_std_delta_nonpos": res.get("n_aopc_std_delta_nonpos"),
            "metrics/n_delta_pos_samples": res.get("n_delta_pos_samples"),
            "metrics/n_delta_nonpos_samples": res.get("n_delta_nonpos_samples"),
            "metrics/gap_n_aopc_mean": res.get("gap_n_aopc_mean"),
            "metrics/gap_n_aopc_std": res.get("gap_n_aopc_std"),
            "metrics/comp_m_mean": res.get("comp_m_mean"),
            "metrics/comp_m_std": res.get("comp_m_std"),
            "metrics/comp_m_primary_mean": res.get("comp_m_primary_mean"),
            "metrics/comp_m_mean_delta_pos": res.get("comp_m_mean_delta_pos"),
            "metrics/comp_m_std_delta_pos": res.get("comp_m_std_delta_pos"),
            "metrics/comp_m_mean_delta_nonpos": res.get("comp_m_mean_delta_nonpos"),
            "metrics/comp_m_std_delta_nonpos": res.get("comp_m_std_delta_nonpos"),
            "metrics/suff_m_mean": res.get("suff_m_mean"),
            "metrics/suff_m_std": res.get("suff_m_std"),
            "metrics/n_comp_m_mean": res.get("n_comp_m_mean"),
            "metrics/n_comp_m_std": res.get("n_comp_m_std"),
            "metrics/n_suff_m_mean": res.get("n_suff_m_mean"),
            "metrics/n_suff_m_std": res.get("n_suff_m_std"),
            "metrics/comp_at_m_mean": res.get("comp_at_m_mean"),
            "metrics/comp_at_m_std": res.get("comp_at_m_std"),
            "metrics/suff_at_m_mean": res.get("suff_at_m_mean"),
            "metrics/suff_at_m_std": res.get("suff_at_m_std"),
            "metrics/delta_max_mean": res.get("delta_max_mean"),
            "metrics/delta_max_std": res.get("delta_max_std"),
            "metrics/delta_max_nonpos_frac": res.get("delta_max_nonpos_frac"),
        }
        # Geometric metrics
        raer = res.get("raer")
        if isinstance(raer, dict):
            log_dict["metrics/raer"] = raer.get("rare_energy_ratio")
        hsa = res.get("hidden_space_alignment")
        if isinstance(hsa, dict):
            log_dict["metrics/hidden_space_alignment"] = hsa.get("hidden_space_alignment")
        # FT metrics (only log if computed, i.e., not None)
        if res.get("ft_mean") is not None:
            log_dict["metrics/ft_mean"] = res.get("ft_mean")
            log_dict["metrics/ft_std"] = res.get("ft_std")
        log_dict = {k: v for k, v in log_dict.items() if v is not None}
        if log_dict:
            wandb.log(log_dict)
    
    # (i) Fixed baseline
    if 'fixed' in baselines:
        print("\n" + "="*80)
        print("Evaluating Fixed baseline")
        print("="*80)
        explainer_fixed = explainer_id
        results['fixed'] = evaluate_baseline_fixed(
            explainer_fixed, model, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode, batch_id=eval_tokens_list_id if compute_ft else None,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
            faith_empty_mode=faith_empty_mode,
            faith_rank_mode=faith_rank_mode,
            faith_rank_top_f=faith_rank_top_f,
            faith_hook_site_mode=faith_hook_site_mode,
        )
        _add_geometric_metrics(results['fixed'], explainer_fixed, id_activations, ood_activations, explainer_type)
        _print_metric_summary("Fixed", results['fixed'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_id,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_id),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['fixed']['diagnostics'] = diag
        print(
            f"[Diagnostics][fixed] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/fixed_auc': results['fixed']['mean_auc']})
            _log_metrics_to_wandb("fixed", results['fixed'])
            _log_diagnostics_to_wandb("fixed", diag)
    

    # (i-b) Fixed + decoder refit (ablation: decoder refit only, no encoder rotation)
    if 'fixed_refit' in baselines:
        print("\n" + "="*80)
        print("Evaluating Fixed + Decoder Refit baseline")
        print("="*80)
        import copy
        explainer_fr = copy.deepcopy(explainer_id)
        with torch.no_grad():
            B_fr = extract_dictionary_from_explainer(explainer_fr, id_activations[:Config.N_COV], explainer_type)
            ood_act_fr = ood_activations[:Config.N_COV].to(B_fr.device)
            A_ood_fr = torch.relu(ood_act_fr @ B_fr)
            refit_device = torch.device(getattr(Config, "GAE_DECODER_REFIT_DEVICE", "cpu"))
            D_fr = decoder_only_refit(
                ood_act_fr.to(refit_device),
                A_ood_fr.to(refit_device),
                lam=float(getattr(Config, "GAE_DECODER_REFIT_LAMBDA", 1e-4)),
            )
            target_dev = B_fr.device
            explainer_fr.D_gae = D_fr.to(target_dev)
            explainer_fr.use_gae = True
            explainer_fr.B_gae = B_fr.to(target_dev)
        print(f"  Decoder refit applied to Fixed explainer (D shape: {D_fr.shape})")
        results['fixed_refit'] = evaluate_baseline_fixed(
            explainer_fr, model, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode, batch_id=eval_tokens_list_id if compute_ft else None,
            faith_m_values=faith_m_values,
            faith_random_repeats=faith_random_repeats,
            faith_m_star=faith_m_star,
            faith_empty_mode=faith_empty_mode,
            faith_rank_mode=faith_rank_mode,
            faith_rank_top_f=faith_rank_top_f,
            faith_hook_site_mode=faith_hook_site_mode,
        )
        _add_geometric_metrics(results['fixed_refit'], explainer_fr, id_activations, ood_activations, explainer_type)
        _print_metric_summary("Fixed+Refit", results['fixed_refit'])
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/fixed_refit_auc': results['fixed_refit']['mean_auc']})
            _log_metrics_to_wandb("fixed_refit", results['fixed_refit'])
        del explainer_fr, B_fr, A_ood_fr, D_fr

    # (ii) OOD-retrained baseline
    if 'ood_retrained' in baselines:
        print("\n" + "="*80)
        print("Evaluating OOD-retrained baseline")
        print("="*80)
        explainer_ood = evaluate_baseline_ood_retrained(
            model, ood_activations[:Config.N_COV], ood_logits[:Config.N_COV],
            layer, explainer_type, d_model, vocab_size,
            max_epochs=ood_retrained_max_epochs,
            max_steps=ood_retrained_max_steps,
            total_tokens=ood_retrained_total_tokens,
            lr=ood_retrained_lr,
            lambda_sparse=ood_retrained_lambda_sparse,
            objective_type=ood_retrained_objective,
            term_t=ood_retrained_term_t,
            batch_size=ood_retrained_batch_size,
            use_streaming=ood_retrained_streaming,
            ood_dataset=ood_dataset_train,
            ood_text_field=text_field,
            context_size=ood_retrained_context_size,
            input_hook_point=getattr(explainer_id, "input_hook_point", None),
            output_hook_point=getattr(explainer_id, "output_hook_point", None),
            warm_start=ood_retrained_warm_start,
            reinit_b_dec=ood_retrained_reinit_b_dec,
            warm_start_explainer=explainer_id,
            use_saelens=ood_retrained_use_saelens,
            use_wandb=use_wandb,
            output_kind=getattr(explainer_id, "output_kind", None),
            task_name="timeshift_ood",
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            retrained_checkpoint_path=ood_retrained_checkpoint_path,
            retrained_checkpoint_dir=ood_retrained_checkpoint_dir,
        )
        results['ood_retrained'] = evaluate_causal_faithfulness(
            model, explainer_ood, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode, batch_id=eval_tokens_list_id if compute_ft else None,
            faith_m_values=faith_m_values, faith_random_repeats=faith_random_repeats, faith_m_star=faith_m_star,
            faith_empty_mode=faith_empty_mode, faith_rank_mode=faith_rank_mode, faith_rank_top_f=faith_rank_top_f,
            faith_hook_site_mode=faith_hook_site_mode,
        )
        _add_geometric_metrics(results['ood_retrained'], explainer_ood, id_activations, ood_activations, explainer_type)
        _print_metric_summary("OOD-retrained", results['ood_retrained'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_ood,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_ood),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['ood_retrained']['diagnostics'] = diag
        print(
            f"[Diagnostics][ood_retrained] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/ood_retrained_auc': results['ood_retrained']['mean_auc']})
            _log_metrics_to_wandb("ood_retrained", results['ood_retrained'])
            _log_diagnostics_to_wandb("ood_retrained", diag)

    # OOD-Finetuned baseline (Kissane 2024 recipe — warm-start fine-tune)
    if 'ood_finetuned' in baselines:
        print("\n" + "="*80)
        print("Evaluating OOD-Finetuned baseline")
        print("="*80)
        explainer_ft = evaluate_baseline_ood_finetuned(
            model, ood_activations[:Config.N_COV], ood_logits[:Config.N_COV],
            layer, explainer_type, d_model, vocab_size,
            total_tokens=ood_finetuned_total_tokens,
            lr=ood_finetuned_lr,
            lambda_sparse=ood_finetuned_lambda_sparse,
            batch_size=ood_finetuned_batch_size,
            context_size=ood_finetuned_context_size,
            loss_type=ood_finetuned_loss_type,
            ood_dataset=ood_dataset_train,
            ood_text_field=text_field,
            ood_dataset_path=None,
            ood_dataset_name=None,
            input_hook_point=getattr(explainer_id, "input_hook_point", None),
            output_hook_point=getattr(explainer_id, "output_hook_point", None),
            warm_start_explainer=explainer_id,
            warm_start_path=explainer_checkpoint_path,
            use_wandb=use_wandb,
            output_kind=getattr(explainer_id, "output_kind", None),
            task_name="timeshift_ood",
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            finetuned_checkpoint_path=ood_finetuned_checkpoint_path,
            finetuned_checkpoint_dir=ood_finetuned_checkpoint_dir,
        )
        results['ood_finetuned'] = evaluate_causal_faithfulness(
            model, explainer_ft, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode, batch_id=eval_tokens_list_id if compute_ft else None,
            faith_m_values=faith_m_values, faith_random_repeats=faith_random_repeats, faith_m_star=faith_m_star,
            faith_empty_mode=faith_empty_mode, faith_rank_mode=faith_rank_mode, faith_rank_top_f=faith_rank_top_f,
            faith_hook_site_mode=faith_hook_site_mode,
        )
        _add_geometric_metrics(results['ood_finetuned'], explainer_ft, id_activations, ood_activations, explainer_type)
        _print_metric_summary("OOD-Finetuned", results['ood_finetuned'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_ft,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_ft),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['ood_finetuned']['diagnostics'] = diag
        print(
            f"[Diagnostics][ood_finetuned] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/ood_finetuned_auc': results['ood_finetuned']['mean_auc']})
            _log_metrics_to_wandb("ood_finetuned", results['ood_finetuned'])
            _log_diagnostics_to_wandb("ood_finetuned", diag)

    # (iii) TERM baseline
    if 'term' in baselines:
        print("\n" + "="*80)
        print("Evaluating TERM baseline")
        print("="*80)
        term_source_path = term_checkpoint_path or explainer_checkpoint_path
        explainer_term = evaluate_baseline_term(
            term_source_path,
            explainer_type,
            d_model,
            vocab_size
        )
        results['term'] = evaluate_causal_faithfulness(
            model, explainer_term, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode, batch_id=eval_tokens_list_id if compute_ft else None,
            faith_m_values=faith_m_values, faith_random_repeats=faith_random_repeats, faith_m_star=faith_m_star,
            faith_empty_mode=faith_empty_mode, faith_rank_mode=faith_rank_mode, faith_rank_top_f=faith_rank_top_f,
            faith_hook_site_mode=faith_hook_site_mode,
        )
        _add_geometric_metrics(results['term'], explainer_term, id_activations, ood_activations, explainer_type)
        _print_metric_summary("TERM", results['term'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_term,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_term),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['term']['diagnostics'] = diag
        print(
            f"[Diagnostics][term] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/term_auc': results['term']['mean_auc']})
            _log_metrics_to_wandb("term", results['term'])
            _log_diagnostics_to_wandb("term", diag)

    # (iv) SAEBoost baseline
    if 'saeboost' in baselines:
        print("\n" + "="*80)
        print("Evaluating SAEBoost baseline")
        print("="*80)
        explainer_saeboost = evaluate_baseline_saeboost(
            base_explainer=explainer_id,
            model=model,
            layer=layer,
            ood_dataset=ood_dataset_train,
            ood_text_field=text_field,
            model_type=explainer_type,
            d_model=d_model,
            vocab_size=vocab_size,
            task_name="timeshift_ood",
            model_name=model_name,
            ood_set=ood_set,
            base_checkpoint_path=explainer_checkpoint_path,
            residual_checkpoint_path=saeboost_residual_checkpoint,
            residual_checkpoint_dir=saeboost_residual_dir,
            resid_coef=saeboost_coef,
            train_residual_if_missing=saeboost_train_residual,
            residual_train_max_epochs=saeboost_residual_max_epochs,
            residual_train_max_steps=saeboost_residual_max_steps,
            residual_train_total_tokens=saeboost_residual_total_tokens,
            residual_train_lr=saeboost_residual_lr,
            residual_train_lambda_sparse=saeboost_residual_lambda_sparse,
            residual_train_objective=saeboost_residual_objective,
            residual_train_term_t=saeboost_residual_term_t,
            residual_train_batch_size=saeboost_residual_batch_size,
            residual_train_context_size=saeboost_residual_context_size,
            residual_train_dict_size=saeboost_residual_dict_size,
            residual_train_reinit_b_dec=saeboost_residual_reinit_b_dec,
            residual_train_sparse_method=saeboost_residual_sparse_method,
            residual_train_batch_top_k=saeboost_residual_batch_top_k,
            residual_train_batch_top_k_start=saeboost_residual_batch_top_k_start,
            residual_train_batch_top_k_warmup_fraction=saeboost_residual_batch_top_k_warmup_fraction,
            base_inference_sparse_method=saeboost_base_infer_sparse_method,
            base_inference_batch_top_k=saeboost_base_infer_batch_top_k,
            base_jumprelu_threshold=saeboost_base_jumprelu_threshold,
            residual_inference_sparse_method=saeboost_residual_infer_sparse_method,
            residual_inference_batch_top_k=saeboost_residual_infer_batch_top_k,
            residual_jumprelu_threshold=saeboost_residual_jumprelu_threshold,
            use_wandb=use_wandb,
        )
        results['saeboost'] = evaluate_causal_faithfulness(
            model, explainer_saeboost, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode, batch_id=eval_tokens_list_id if compute_ft else None,
            faith_m_values=faith_m_values, faith_random_repeats=faith_random_repeats, faith_m_star=faith_m_star,
            faith_empty_mode=faith_empty_mode, faith_rank_mode=faith_rank_mode, faith_rank_top_f=faith_rank_top_f,
            faith_hook_site_mode=faith_hook_site_mode,
        )
        _add_geometric_metrics(results['saeboost'], explainer_saeboost, id_activations, ood_activations, explainer_type)
        _print_metric_summary("SAEBoost", results['saeboost'])
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_saeboost,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_saeboost),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['saeboost']['diagnostics'] = diag
        print(
            f"[Diagnostics][saeboost] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/saeboost_auc': results['saeboost']['mean_auc']})
            _log_metrics_to_wandb("saeboost", results['saeboost'])
            _log_diagnostics_to_wandb("saeboost", diag)
    

    # (iv-b) FaithfulSAE baseline
    if 'faithfulsae' in baselines:
        faithfulsae_ckpt = faithfulsae_checkpoint_path
        if faithfulsae_ckpt is None:
            print("WARNING: --faithfulsae_checkpoint not provided. Skipping FaithfulSAE baseline.")
        else:
            print("\n" + "="*80)
            print("Evaluating FaithfulSAE baseline")
            print("="*80)
            explainer_faithfulsae = evaluate_baseline_faithfulsae(
                faithfulsae_ckpt,
                explainer_type,
                d_model,
                vocab_size
            )
            results['faithfulsae'] = evaluate_causal_faithfulness(
                model, explainer_faithfulsae, eval_tokens_list, layer, explainer_type,
                target_mode=target_mode, batch_id=eval_tokens_list_id if compute_ft else None,
                faith_m_values=faith_m_values, faith_random_repeats=faith_random_repeats, faith_m_star=faith_m_star,
                faith_empty_mode=faith_empty_mode, faith_rank_mode=faith_rank_mode, faith_rank_top_f=faith_rank_top_f,
                faith_hook_site_mode=faith_hook_site_mode,
            )
            _add_geometric_metrics(results['faithfulsae'], explainer_faithfulsae, id_activations, ood_activations, explainer_type)
            _print_metric_summary("FaithfulSAE", results['faithfulsae'])
            # Diagnostics
            diag = compute_ood_diagnostics(
                model=model,
                explainer=explainer_faithfulsae,
                tokens_list=eval_tokens_list,
                layer=layer,
                model_type=explainer_type,
                **_diagnostics_kwargs(explainer_faithfulsae),
                batch_size=DIAG_BATCH_SIZE,
                max_samples=DIAG_MAX_SAMPLES,
            )
            results['faithfulsae']['diagnostics'] = diag
            print(
                f"[Diagnostics][faithfulsae] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
                f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
            )
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({'baseline/faithfulsae_auc': results['faithfulsae']['mean_auc']})
                _log_metrics_to_wandb("faithfulsae", results['faithfulsae'])
                _log_diagnostics_to_wandb("faithfulsae", diag)

    # (v) GAE baseline
    if 'gae' in baselines:
        print("\n" + "="*80)
        print("Evaluating GAE baseline")
        print("="*80)
        _gae_ce_w = _compute_gae_ce_weights(model, eval_tokens_list, layer, explainer_id)
        explainer_gae = evaluate_baseline_gae(
            explainer_id,
            model,
            id_activations,
            ood_activations[:Config.N_COV],
            id_activations_out=id_logits[:Config.N_COV] if id_logits is not None else None,
            ood_activations_out=ood_activations_out[:Config.N_COV] if ood_activations_out is not None else None,
            layer=layer,
            model_type=explainer_type,
            r=gae_r,
            rank_mode=gae_rank_mode,
            rank_energy=gae_rank_energy,
            rank_delta_mode=gae_rank_delta_mode,
            rank_min=gae_rank_min,
            rank_max=gae_rank_max,
            decoder_mode=gae_decoder_mode,
            dict_space=gae_dict_space,
            recon_lambda=gae_recon_lambda,
            ce_weights=_gae_ce_w,
        )
        results['gae'] = evaluate_causal_faithfulness(
            model, explainer_gae, eval_tokens_list, layer, explainer_type,
            target_mode=target_mode, batch_id=eval_tokens_list_id if compute_ft else None,
            faith_m_values=faith_m_values, faith_random_repeats=faith_random_repeats, faith_m_star=faith_m_star,
            faith_empty_mode=faith_empty_mode, faith_rank_mode=faith_rank_mode, faith_rank_top_f=faith_rank_top_f,
            faith_hook_site_mode=faith_hook_site_mode,
        )
        _add_geometric_metrics(results['gae'], explainer_gae, id_activations, ood_activations, explainer_type)
        results['gae']['gae_metadata'] = {
            'decoder_mode': getattr(explainer_gae, 'gae_decoder_mode', None),
            'rank_mode': getattr(explainer_gae, 'gae_rank_mode', None),
            'rank_eff': getattr(explainer_gae, 'gae_rank_eff', None),
            'geometry_space': getattr(explainer_gae, 'gae_geometry_space', None),
            'geometry_source': getattr(explainer_gae, 'gae_geometry_source', None),
            'dict_space': getattr(explainer_gae, 'gae_dict_space', None),
        }
        _print_metric_summary("GAE", results['gae'])
        # Diagnostics
        diag = compute_ood_diagnostics(
            model=model,
            explainer=explainer_gae,
            tokens_list=eval_tokens_list,
            layer=layer,
            model_type=explainer_type,
            **_diagnostics_kwargs(explainer_gae),
            batch_size=DIAG_BATCH_SIZE,
            max_samples=DIAG_MAX_SAMPLES,
        )
        results['gae']['diagnostics'] = diag
        print(
            f"[Diagnostics][gae] ReconErr={diag['recon_err_mean']:.4g}±{diag['recon_err_std']:.4g} | "
            f"LogitErr0={diag['logit_err0_mean']:.4g}±{diag['logit_err0_std']:.4g}"
        )
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'baseline/gae_auc': results['gae']['mean_auc']})
            _log_metrics_to_wandb("gae", results['gae'])
            _log_diagnostics_to_wandb("gae", diag)
    
    # Save results
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build results summary with all metrics for evaluated baselines
    results_summary = _build_results_summary(
        task_name='timeshift_ood',
        model_name=model_name,
        explainer_type=explainer_type,
        ood_set=ood_set,
        layer=layer,
        baselines=baselines,
        explainer=explainer_id,
        config_extra={
            'target_mode': target_mode,
            'compute_ft': bool(compute_ft),
            'gae_r': int(gae_r),
            'gae_rank_mode': gae_rank_mode,
            'gae_rank_energy': float(gae_rank_energy),
            'gae_rank_delta_mode': gae_rank_delta_mode,
            'gae_rank_min': int(gae_rank_min),
            'gae_rank_max': int(gae_rank_max) if gae_rank_max is not None else None,
            'gae_decoder_mode': gae_decoder_mode,
            'gae_dict_space': gae_dict_space,
            'gae_recon_lambda': float(gae_recon_lambda),
            'n_eval_samples': int(n_eval_samples),
            'faith_empty_mode': faith_empty_mode,
            'faith_rank_mode': faith_rank_mode,
            'faith_rank_top_f': int(faith_rank_top_f),
            'faith_hook_site_mode': faith_hook_site_mode,
            'timeshift_span_mode': timeshift_span_mode,
        },
    )
    
    # Add full metrics for evaluated baselines
    for baseline in baselines:
        if baseline in results:
            results_summary['baselines'][baseline] = results[baseline]
    
    # Log summary to wandb (only for evaluated baselines)
    if use_wandb and WANDB_AVAILABLE:
        wandb_summary = {}
        for baseline in baselines:
            if baseline in results:
                wandb_summary[f'baseline/{baseline}_auc'] = results[baseline]['mean_auc']
        wandb.summary.update(wandb_summary)
        wandb.finish()
    
    # Compute Delta CE if requested
    # Compute Delta CE for all baselines
    _explainer_map = {}
    if 'fixed' in baselines:
        _explainer_map['fixed'] = explainer_id
    if 'term' in baselines and 'explainer_term' in locals():
        _explainer_map['term'] = explainer_term
    if 'saeboost' in baselines and 'explainer_saeboost' in locals():
        _explainer_map['saeboost'] = explainer_saeboost
    if 'gae' in baselines and 'explainer_gae' in locals():
        _explainer_map['gae'] = explainer_gae
    if 'faithfulsae' in baselines and 'explainer_faithfulsae' in locals():
        _explainer_map['faithfulsae'] = explainer_faithfulsae
    if 'ood_retrained' in baselines and 'explainer_ood' in locals():
        _explainer_map['ood_retrained'] = explainer_ood
    if 'ood_finetuned' in baselines and 'explainer_ft' in locals():
        _explainer_map['ood_finetuned'] = explainer_ft
    _compute_delta_ce_for_all_baselines(
        results, _explainer_map, model, eval_tokens_list, layer, explainer_type,
    )
    for bl_name in _explainer_map:
        if bl_name in results and bl_name in results_summary.get('baselines', {}):
            results_summary['baselines'][bl_name] = results[bl_name]

    # Create filename with baseline info
    baseline_str = '_'.join(sorted(baselines))
    results_file = os.path.join(
        output_dir,
        f"timeshift_ood_{model_name}_{explainer_type}_{ood_set}_{baseline_str}.json"
    )

    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return results_summary


def run_faithfulness_metrics(
    jsonl_path: str,
    model_name: str,
    explainer_type: str,
    explainer_checkpoint_path: str,
    out_csv: str,
    out_summary: str,
    m_list: List[int],
    random_repeats: int = 5,
    dataset_filter: str = "all",
    m_star: int = 32,
    seed: int = 0,
    n_eval: int = -1,
):
    """
    Compute AOPC / nAOPC and Comp/Suff metrics from a JSONL file of prompts.

    This uses the SAME feature ablation scoring logic as evaluate_causal_faithfulness
    (via compute_feature_mask_scores).
    """
    if not jsonl_path:
        raise ValueError("jsonl_path is required")

    set_seed(seed)

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    records = _load_domain_ood_jsonl_records(jsonl_path)
    if dataset_filter and dataset_filter != "all":
        records = [r for r in records if r.get("dataset") == dataset_filter]

    if n_eval is not None and n_eval > 0:
        records = records[:n_eval]

    if len(records) == 0:
        raise ValueError("No records to evaluate after filtering.")

    # Load model / explainer
    model, tokenizer = load_model(model_name)
    layer = get_layer(model_name)
    if layer is None:
        raise ValueError(f"Unknown model name '{model_name}' for layer selection.")

    d_model = model.cfg.d_model
    vocab_size = model.cfg.d_vocab
    explainer = load_explainer(explainer_checkpoint_path, explainer_type, d_model, vocab_size=vocab_size)

    device = Config.device
    model.to(device)
    if isinstance(explainer, EXPLAINER_TYPES):
        explainer.to(device)

    tokens_list, target_token_ids = _records_to_tokens(records)

    # Prepare output paths
    out_csv = str(out_csv)
    if out_summary is None:
        out_summary = out_csv + ".summary.json"

    # Compute metrics per example
    rows = []
    agg = {
        "aopc": [],
        "aopc_rand": [],
        "gap_aopc": [],
        "n_aopc": [],
        "gap_n_aopc": [],
        "comp": [],
        "suff": [],
        "n_comp": [],
        "n_suff": [],
        "comp_at_m": [],
        "suff_at_m": [],
        "delta_max": [],
    }

    for rec, tokens, target_id in zip(records, tokens_list, target_token_ids):
        tokens = tokens.to(device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        # Skip invalid target ids
        if target_id < 0 or target_id >= vocab_size:
            continue

        h = extract_residual_stream_activations(model, tokens, layer, position=-1)
        logits = extract_logits(model, tokens, position=-1)

        scores = compute_feature_mask_scores(
            explainer=explainer,
            model=model,
            tokens=tokens,
            layer=layer,
            position=-1,
            h=h,
            logits_original=logits,
            target_token_idx=target_id,
            model_type=explainer_type,
            m_values=m_list,
            random_repeats=random_repeats,
        )
        aopc = compute_aopc_and_normalized(scores, m_list)
        comp = compute_comp_suff(scores, m_list, m_star=m_star)

        target_str = tokenizer.decode([target_id])
        row = {
            "dataset": rec.get("dataset"),
            "target_id": int(target_id),
            "target_str": target_str,
            "k": int(scores["k"]),
            "s0": float(scores["s0"]),
            "s_empty": float(scores["s_empty"]),
            "aopc": aopc["aopc"],
            "aopc_rand": aopc["aopc_rand"],
            "gap_aopc": aopc["gap_aopc"],
            "n_aopc": aopc["n_aopc"],
            "gap_n_aopc": aopc["gap_n_aopc"],
            "comp": comp["comp"],
            "suff": comp["suff"],
            "n_comp": comp["n_comp"],
            "n_suff": comp["n_suff"],
            "comp_at_m": comp["comp_at_m"],
            "suff_at_m": comp["suff_at_m"],
            "delta_max": comp["delta_max"],
            "flag_delta_max_nonpos": aopc["flag_delta_max_nonpos"],
        }

        rows.append(row)
        for k in agg.keys():
            agg[k].append(row[k])

    # Write CSV
    import csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    else:
        print("Warning: no rows to write.")

    # Summary stats
    summary = {
        "n_examples": len(rows),
        "model": model_name,
        "explainer": explainer_type,
        "jsonl": str(jsonl_path),
        "dataset_filter": dataset_filter,
        "m_list": m_list,
        "random_repeats": random_repeats,
        "m_star": m_star,
    }
    for k, vals in agg.items():
        if len(vals) == 0:
            summary[f"{k}_mean"] = None
            summary[f"{k}_std"] = None
        else:
            arr = np.array(vals, dtype=np.float64)
            summary[f"{k}_mean"] = float(arr.mean())
            summary[f"{k}_std"] = float(arr.std())

    os.makedirs(os.path.dirname(out_summary) or ".", exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Faithfulness] Wrote {len(rows)} rows to {out_csv}")
    print(f"[Faithfulness] Summary saved to {out_summary}")
    return summary


def main():
    """Unified OOD experiment runner."""
    parser = argparse.ArgumentParser(description='Run OOD experiments')
    subparsers = parser.add_subparsers(dest='task', required=True)

    # -----------------------------
    # Argument helper functions
    # -----------------------------

    def add_global_common_args(p):
        """Arguments shared across all baselines."""
        p.add_argument(
            '--model',
            type=str,
            required=True,
            choices=['gpt2', 'pythia-410m', 'pythia-1.4b'],
            help='Model to use',
        )
        p.add_argument(
            '--explainer',
            type=str,
            required=True,
            choices=['transcoder', 'sae'],
            help='Explainer type',
        )
        p.add_argument(
            '--checkpoint',
            type=str,
            required=True,
            help='Path to ID-trained explainer checkpoint',
        )
        p.add_argument(
            '--output_dir',
            type=str,
            default=None,
            help='Output directory for results',
        )
        p.add_argument(
            '--n_eval',
            type=int,
            default=1000,
            help='Number of evaluation samples (-1 to use all available samples)',
        )
        p.add_argument(
            '--batch_size',
            type=int,
            default=None,
            help='Override Config.BATCH_SIZE for evaluation/activation collection',
        )
        p.add_argument(
            '--force_recompute',
            action='store_true',
            help='Force recomputation of activations (ignore cache)',
        )
        p.add_argument(
            '--use_wandb',
            action='store_true',
            help='Use wandb for logging',
        )
        p.add_argument(
            '--wandb_project',
            type=str,
            default=None,
            help='Wandb project name',
        )
        p.add_argument(
            '--wandb_name',
            type=str,
            default=None,
            help='Wandb run name (default: auto-generated)',
        )
        p.add_argument(
            '--baselines',
            type=str,
            nargs='+',
            choices=['fixed', 'fixed_refit', 'ood_retrained', 'ood_finetuned', 'term', 'gae', 'saeboost', 'faithfulsae'],
            default=None,
            help=(
                'Baselines to evaluate (default: fixed ood_retrained term gae). '
                'Can specify multiple: --baselines fixed gae'
            ),
        )
        p.add_argument(
            '--saelens_store_batch_size',
            type=int,
            default=None,
            help='Override SAELens activation store batch size (lower to reduce memory)',
        )
        p.add_argument(
            '--saelens_n_batches_in_buffer',
            type=int,
            default=None,
            help='Override SAELens n_batches_in_buffer (lower to reduce memory)',
        )
        p.add_argument(
            '--target_mode',
            type=str,
            default='argmax',
            choices=['argmax', 'gold', 'provided'],
            help='Target token selection mode: argmax, gold, or provided (dataset-derived)',
        )
        p.add_argument(
            '--compute_ft',
            action='store_true',
            help='Compute Faithfulness Transfer (requires ID batch)',
        )
        p.add_argument(
            '--faith_m',
            type=str,
            default="1,2,4,8,16,32,64,128",
            help='Comma-separated M values for AOPC/Comp/Suff (empty to disable)',
        )
        p.add_argument(
            '--faith_random_repeats',
            type=int,
            default=3,
            help='Random repeats for AOPC baseline',
        )
        p.add_argument(
            '--faith_m_star',
            type=int,
            default=32,
            help='m* for Comp@M and Suff@M',
        )
        p.add_argument(
            '--faith_empty_mode',
            type=str,
            default='zero_resid',
            choices=['zero_resid', 'decoder_bias'],
            help='Empty-feature handling mode for faithfulness metrics.',
        )
        p.add_argument(
            '--faith_rank_mode',
            type=str,
            default='causal_topz',
            choices=['activation', 'causal_topz'],
            help='Feature ranking mode used by faithfulness metrics.',
        )
        p.add_argument(
            '--faith_rank_top_f',
            type=int,
            default=64,
            help='Top-F features to prefilter before causal ranking (ignored for activation ranking).',
        )
        p.add_argument(
            '--faith_hook_site_mode',
            type=str,
            default='auto',
            choices=list(FAITH_HOOK_SITE_MODES),
            help='Hook site mode used for faithfulness intervention (Issue 10 diagnostics).',
        )

    def add_fixed_args(p):
        """Fixed baseline currently has no extra arguments."""
        return p

    def add_ood_retrained_args(p):
        p.add_argument(
            '--ood_retrained_max_epochs',
            type=int,
            default=50,
            help='Max epochs for OOD-retrained baseline',
        )
        p.add_argument(
            '--ood_retrained_max_steps',
            type=int,
            default=None,
            help='Max training steps for OOD-retrained baseline (overrides max_epochs if set)',
        )
        p.add_argument(
            '--ood_retrained_total_tokens',
            type=int,
            default=None,
            help='Total training tokens for OOD-retrained baseline (overrides max_steps/max_epochs if set)',
        )
        p.add_argument(
            '--ood_retrained_lr',
            type=float,
            default=1e-3,
            help='Learning rate for OOD-retrained baseline',
        )
        p.add_argument(
            '--ood_retrained_lambda_sparse',
            type=float,
            default=0.01,
            help='L1 regularization for OOD-retrained baseline',
        )
        p.add_argument(
            '--ood_retrained_objective',
            type=str,
            default='ERM',
            choices=['ERM', 'TERM'],
            help='Training objective for OOD-retrained baseline',
        )
        p.add_argument(
            '--ood_retrained_term_t',
            type=float,
            default=None,
            help='TERM tilt parameter for OOD-retrained baseline (used when objective=TERM)',
        )
        p.add_argument(
            '--ood_retrained_batch_size',
            type=int,
            default=None,
            help='Training batch size for OOD-retrained baseline (default: Config.BATCH_SIZE)',
        )
        p.add_argument(
            '--ood_retrained_streaming',
            action='store_true',
            help='Use streaming OOD data for retraining (train_explainers-style)',
        )
        p.add_argument(
            '--ood_retrained_context_size',
            type=int,
            default=128,
            help='Context size for streaming OOD retraining',
        )
        p.add_argument(
            '--ood_retrained_warm_start',
            dest='ood_retrained_warm_start',
            action='store_true',
            default=True,
            help='Warm-start OOD retraining from ID explainer weights (default: True)',
        )
        p.add_argument(
            '--no_ood_retrained_warm_start',
            dest='ood_retrained_warm_start',
            action='store_false',
            help='Disable warm-start for OOD retraining',
        )
        p.add_argument(
            '--ood_retrained_reinit_b_dec',
            dest='ood_retrained_reinit_b_dec',
            action='store_true',
            default=True,
            help='Reinitialize b_dec at start of OOD retraining (default: True)',
        )
        p.add_argument(
            '--no_ood_retrained_reinit_b_dec',
            dest='ood_retrained_reinit_b_dec',
            action='store_false',
            help='Disable b_dec reinitialization for OOD retraining',
        )
        p.add_argument(
            '--ood_retrained_use_saelens',
            dest='ood_retrained_use_saelens',
            action='store_true',
            default=True,
            help='Use SAELens model for OOD retraining (default: True)',
        )
        p.add_argument(
            '--no_ood_retrained_use_saelens',
            dest='ood_retrained_use_saelens',
            action='store_false',
            help='Disable SAELens OOD retraining (not supported)',
        )
        p.add_argument(
            '--ood_retrained_checkpoint',
            type=str,
            default=None,
            help='Optional checkpoint path for cached OOD-retrained explainer. If it exists, load instead of retraining; otherwise train and save there.',
        )
        p.add_argument(
            '--ood_retrained_checkpoint_dir',
            type=str,
            default=os.environ.get('REPO_DATA', './data') + '/checkpoints',
            help='Directory used for automatic OOD-retrained checkpoint caching. If this already points to a task folder such as checkpoints/timeshift_ood, checkpoints are stored directly there.',
        )
        return p

    def add_ood_finetuned_args(p):
        # OOD-Finetuned baseline (Kissane 2024 recipe).
        # warm_start and reinit_b_dec are intentionally not exposed: the Kissane
        # recipe hard-codes them (warm_start=True, reinit_b_dec=False) in
        # evaluate_baseline_ood_finetuned.
        p.add_argument(
            '--ood_finetuned_total_tokens',
            type=int,
            default=5_000_000,
            help='Total OOD tokens for ood_finetuned fine-tune (Kissane default: 5M)',
        )
        p.add_argument(
            '--ood_finetuned_lr',
            type=float,
            default=None,
            help='Learning rate for ood_finetuned (default: match ID checkpoint LR)',
        )
        p.add_argument(
            '--ood_finetuned_lambda_sparse',
            type=float,
            default=None,
            help='L1 coefficient for ood_finetuned (default: match ID checkpoint)',
        )
        p.add_argument(
            '--ood_finetuned_batch_size',
            type=int,
            default=256,
            help='Training batch size for ood_finetuned (Kissane default: 256)',
        )
        p.add_argument(
            '--ood_finetuned_context_size',
            type=int,
            default=128,
            help='Context size for streaming OOD fine-tune',
        )
        p.add_argument(
            '--ood_finetuned_loss_type',
            type=str,
            default='mse',
            choices=['mse', 'kl'],
            help='Loss type: mse (Variant A, Kissane) or kl (Variant B, not implemented)',
        )
        p.add_argument(
            '--ood_finetuned_checkpoint_path',
            type=str,
            default=None,
            help='Explicit checkpoint save path for ood_finetuned (optional)',
        )
        p.add_argument(
            '--ood_finetuned_checkpoint_dir',
            type=str,
            default=os.environ.get('REPO_DATA', './data') + '/checkpoints/ood_finetuned',
            help='Checkpoint root directory for ood_finetuned',
        )
        return p

    def add_term_args(p):
        p.add_argument(
            '--term_checkpoint',
            type=str,
            default=None,
            help='Optional TERM explainer checkpoint override. If omitted, --checkpoint is evaluated as TERM.',
        )
        return p

    def add_faithfulsae_args(p):
        p.add_argument(
            '--faithfulsae_checkpoint',
            type=str,
            default=None,
            help='Path to FaithfulSAE-trained explainer checkpoint',
        )
        return p

    def add_gae_args(p):
        p.add_argument(
            '--gae_r',
            type=int,
            default=64,
            help='GAE subspace rank',
        )
        p.add_argument(
            '--gae_rank_mode',
            type=str,
            default='fixed',
            choices=['fixed', 'energy'],
            help="GAE rank selection mode: 'fixed' uses --gae_r, 'energy' auto-selects rank from Delta-Cov.",
        )
        p.add_argument(
            '--gae_rank_energy',
            type=float,
            default=0.99,
            help='Target cumulative energy ratio for --gae_rank_mode energy.',
        )
        p.add_argument(
            '--gae_rank_delta_mode',
            type=str,
            default='ood',
            choices=['ood', 'contrastive'],
            help='Delta-Cov mode for energy rank selection.',
        )
        p.add_argument(
            '--gae_rank_min',
            type=int,
            default=1,
            help='Lower bound for auto-selected GAE rank in energy mode.',
        )
        p.add_argument(
            '--gae_rank_max',
            type=int,
            default=None,
            help='Optional upper bound for auto-selected GAE rank in energy mode.',
        )
        p.add_argument(
            '--gae_decoder_mode',
            type=str,
            default='theory',
            choices=['theory', 'affine', 'residual_orig', 'residual_rotated',
                     'encoder_rotation', 'selective_columns', 'diag_gains'],
            help=(
                'GAE decoder mode: theory | affine | residual_orig | residual_rotated | '
                'encoder_rotation | selective_columns. '
                'residual_orig: W_orig + ΔW (closed-form residual on ID decoder). '
                'residual_rotated: W̃ + ΔW (residual on Procrustes-rotated decoder). theory is the '
                'limit case of affine with lam_gae→∞, lam_id=lam_geom=0, frozen bias. '
                'encoder_rotation (SAE-only): apply Procrustes rotation R* to W_enc, '
                'preserving W_dec/b_dec — for Top-K SAE where decoder columns are rigid. '
                'selective_columns (SAE-only): residual fit on OOD-active decoder columns only; '
                'other columns frozen at W_dec. '
                'Other previously supported modes (sc, diag, mix, consistent, none, '
                'refit) were removed on 2026-04-14 — see docs/gae_260414.md.'
            ),
        )
        p.add_argument(
            '--gae_dict_method',
            type=str,
            default=None,
            choices=['auto', 'normal', 'row'],
            help='GAE dictionary extraction solver (overrides Config.GAE_DICT_METHOD when provided).',
        )
        p.add_argument(
            '--gae_dict_space',
            type=str,
            default='auto',
            choices=['auto', 'decoder', 'empirical'],
            help='GAE dictionary space: auto/decoder (use W_dec), empirical (legacy h-space LS).',
        )
        p.add_argument(
            '--gae_recon_lambda',
            type=float,
            default=0.0,
            help='Weight for reconstruction term in GAE-R closed-form (0.0 disables).',
        )
        return p

    def add_saeboost_args(p):
        p.add_argument(
            '--saeboost_residual_checkpoint',
            type=str,
            default=None,
            help='Path to SAEBoost residual explainer checkpoint (overrides auto-discovery)',
        )
        p.add_argument(
            '--saeboost_residual_dir',
            type=str,
            default='checkpoints',
            help='Root directory used to auto-discover SAEBoost residual checkpoints',
        )
        p.add_argument(
            '--saeboost_coef',
            type=float,
            default=1.0,
            help='Residual scaling coefficient for SAEBoost (default: 1.0)',
        )
        p.add_argument(
            '--saeboost_train_residual',
            action='store_true',
            help='Train SAEBoost residual explainer when checkpoint is not found',
        )
        p.add_argument(
            '--saeboost_residual_max_epochs',
            type=int,
            default=50,
            help='Max epochs (treated as max_steps for streaming) for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_max_steps',
            type=int,
            default=None,
            help='Max training steps for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_total_tokens',
            type=int,
            default=None,
            help='Total training tokens for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_lr',
            type=float,
            default=8e-4,
            help='Learning rate for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_lambda_sparse',
            type=float,
            default=5e-6,
            help='L1 regularization for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_objective',
            type=str,
            default='ERM',
            choices=['ERM', 'TERM'],
            help='Objective for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_term_t',
            type=float,
            default=None,
            help='TERM tilt parameter for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_batch_size',
            type=int,
            default=None,
            help='Training batch size for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_context_size',
            type=int,
            default=128,
            help='Context size for SAEBoost residual streaming training',
        )
        p.add_argument(
            '--saeboost_residual_dict_size',
            type=int,
            default=1024,
            help='Residual dictionary size target for SAEBoost training',
        )
        p.add_argument(
            '--saeboost_residual_reinit_b_dec',
            dest='saeboost_residual_reinit_b_dec',
            action='store_true',
            default=True,
            help='Reinitialize b_dec at start of SAEBoost residual training (default: True)',
        )
        p.add_argument(
            '--no_saeboost_residual_reinit_b_dec',
            dest='saeboost_residual_reinit_b_dec',
            action='store_false',
            help='Disable b_dec reinitialization for SAEBoost residual training',
        )
        p.add_argument(
            '--saeboost_residual_sparse_method',
            type=str,
            default='l1',
            choices=['l1', 'batchtopk'],
            help='Sparsity method for SAEBoost residual training (default: l1)',
        )
        p.add_argument(
            '--saeboost_residual_batch_top_k',
            type=int,
            default=5,
            help='Per-sample k for BatchTopK residual training',
        )
        p.add_argument(
            '--saeboost_residual_batch_top_k_start',
            type=int,
            default=None,
            help='Initial k for BatchTopK warmup (default: residual dict size)',
        )
        p.add_argument(
            '--saeboost_residual_batch_top_k_warmup_fraction',
            type=float,
            default=0.1,
            help='Warmup fraction for BatchTopK k-annealing',
        )
        p.add_argument(
            '--saeboost_base_infer_sparse_method',
            type=str,
            default='auto',
            choices=['auto', 'none', 'l1', 'batchtopk', 'jumprelu'],
            help='Sparse mode for base explainer during SAEBoost inference',
        )
        p.add_argument(
            '--saeboost_base_infer_batch_top_k',
            type=int,
            default=None,
            help='BatchTopK k for base explainer during SAEBoost inference',
        )
        p.add_argument(
            '--saeboost_base_jumprelu_threshold',
            type=float,
            default=None,
            help='JumpReLU threshold for base explainer during SAEBoost inference',
        )
        p.add_argument(
            '--saeboost_residual_infer_sparse_method',
            type=str,
            default='auto',
            choices=['auto', 'none', 'l1', 'batchtopk', 'jumprelu'],
            help='Sparse mode for residual explainer during SAEBoost inference',
        )
        p.add_argument(
            '--saeboost_residual_infer_batch_top_k',
            type=int,
            default=None,
            help='BatchTopK k for residual explainer during SAEBoost inference',
        )
        p.add_argument(
            '--saeboost_residual_jumprelu_threshold',
            type=float,
            default=None,
            help='JumpReLU threshold for residual explainer during SAEBoost inference',
        )
        return p

    def add_common_args(p):
        """Attach all arguments needed for backwards-compatible CLI."""
        add_global_common_args(p)
        add_fixed_args(p)
        add_ood_retrained_args(p)
        add_ood_finetuned_args(p)
        add_term_args(p)
        add_faithfulsae_args(p)
        add_gae_args(p)
        add_saeboost_args(p)

    adv = subparsers.add_parser('adv', help='Adversarial OOD')
    add_common_args(adv)
    adv.add_argument('--ood_set', type=str, required=True,
                     choices=['halu_eval', 'jailbreakbench', 'jailbreakhub'],
                     help='OOD adversarial dataset')
    adv.add_argument('--min_len_ood', type=int, default=None,
                     help='Override minimum token length for OOD sampling')
    adv.add_argument('--min_len_eval', type=int, default=None,
                     help='Override minimum token length for eval prompts')
    adv.add_argument(
        '--adv_ood_token_budget',
        type=int,
        default=2_000_000,
        help='Token budget used to build adversarial OOD activation pool (default: 2,000,000)',
    )
    adv.add_argument(
        '--adv_bootstrap_repeats',
        type=int,
        default=100,
        help='Bootstrap repetitions for adversarial CI reporting (default: 100)',
    )
    adv.add_argument(
        '--adv_bootstrap_ci',
        type=float,
        default=95.0,
        help='Bootstrap CI level in percent for adversarial reporting (default: 95)',
    )

    dom = subparsers.add_parser('domain', help='Domain OOD')
    add_common_args(dom)
    dom.add_argument('--ood_set', type=str, required=True,
                     choices=['patents', 'edgar', 'govreport', 'standards'],
                     help='OOD domain dataset')
    dom.add_argument('--domain_ood_jsonl_dir', type=str, default="domain_ood_prompts",
                     help='Directory containing per-domain JSONL files (default: domain_ood_prompts)')
    dom.add_argument('--no_domain_ood_jsonl', action='store_true',
                     help='Disable JSONL input and fall back to HF datasets')
    dom.add_argument('--no_domain_ood_jsonl_streaming', action='store_true',
                     help='Disable JSONL streaming retrain (use HF train split instead)')
    dom.add_argument('--domain_ood_force_provided_target', action='store_true',
                     help='Force target_mode=provided when using JSONL inputs')

    ts = subparsers.add_parser('timeshift', help='Time-shift OOD')
    add_common_args(ts)
    ts.add_argument('--ood_set', type=str, required=True,
                    choices=['fineweb', 'dolma_web'],
                    help='OOD time-shift dataset')
    ts.add_argument(
        '--timeshift_span_mode',
        type=str,
        default='prefix',
        choices=['prefix', 'middle'],
        help='Token span extraction mode for Time-shift OOD (use one mode for all models for fair comparison).',
    )

    # -----------------------------
    # Baseline subparsers (optional)
    # -----------------------------

    def add_baseline_subparsers(parent):
        """Create baseline-specific subparsers that only configure arguments."""
        baseline_subparsers = parent.add_subparsers(
            dest='baseline',
            required=False,
            help='Baseline-specific configuration (optional)',
        )

        fixed_p = baseline_subparsers.add_parser('fixed', help='Fixed baseline')
        add_global_common_args(fixed_p)
        add_fixed_args(fixed_p)

        ood_p = baseline_subparsers.add_parser('ood_retrained', help='OOD-retrained baseline')
        add_global_common_args(ood_p)
        add_ood_retrained_args(ood_p)

        term_p = baseline_subparsers.add_parser('term', help='TERM baseline')
        add_global_common_args(term_p)
        add_term_args(term_p)

        gae_p = baseline_subparsers.add_parser('gae', help='GAE baseline')
        add_global_common_args(gae_p)
        add_gae_args(gae_p)

        faithfulsae_p = baseline_subparsers.add_parser('faithfulsae', help='FaithfulSAE baseline')
        add_global_common_args(faithfulsae_p)
        add_faithfulsae_args(faithfulsae_p)

        saeboost_p = baseline_subparsers.add_parser('saeboost', help='SAEBoost baseline')
        add_global_common_args(saeboost_p)
        add_saeboost_args(saeboost_p)

    add_baseline_subparsers(adv)
    add_baseline_subparsers(dom)
    add_baseline_subparsers(ts)

    args = parser.parse_args()

    # Allow "task baseline ..." style to populate --baselines when omitted.
    if getattr(args, 'baseline', None) is not None and args.baselines is None:
        args.baselines = [args.baseline]
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be positive.")
        print(f"Overriding Config.BATCH_SIZE: {Config.BATCH_SIZE} -> {args.batch_size}")
        Config.BATCH_SIZE = args.batch_size
    if args.saelens_store_batch_size is not None:
        if args.saelens_store_batch_size <= 0:
            raise ValueError("--saelens_store_batch_size must be positive.")
        print(
            f"Setting Config.SAELENS_STORE_BATCH_SIZE: {Config.SAELENS_STORE_BATCH_SIZE} -> "
            f"{args.saelens_store_batch_size}"
        )
        Config.SAELENS_STORE_BATCH_SIZE = args.saelens_store_batch_size
    if args.saelens_n_batches_in_buffer is not None:
        if args.saelens_n_batches_in_buffer <= 0:
            raise ValueError("--saelens_n_batches_in_buffer must be positive.")
        print(
            f"Setting Config.SAELENS_N_BATCHES_IN_BUFFER: {Config.SAELENS_N_BATCHES_IN_BUFFER} -> "
            f"{args.saelens_n_batches_in_buffer}"
        )
        Config.SAELENS_N_BATCHES_IN_BUFFER = args.saelens_n_batches_in_buffer
    if args.saeboost_residual_batch_top_k <= 0:
        raise ValueError("--saeboost_residual_batch_top_k must be positive.")
    if args.saeboost_residual_batch_top_k_warmup_fraction < 0 or args.saeboost_residual_batch_top_k_warmup_fraction > 1:
        raise ValueError("--saeboost_residual_batch_top_k_warmup_fraction must be in [0, 1].")
    if args.saeboost_base_infer_batch_top_k is not None and args.saeboost_base_infer_batch_top_k <= 0:
        raise ValueError("--saeboost_base_infer_batch_top_k must be positive when provided.")
    if args.saeboost_residual_infer_batch_top_k is not None and args.saeboost_residual_infer_batch_top_k <= 0:
        raise ValueError("--saeboost_residual_infer_batch_top_k must be positive when provided.")
    if (
        args.saeboost_base_infer_sparse_method == "jumprelu"
        and args.saeboost_base_jumprelu_threshold is None
    ):
        raise ValueError(
            "--saeboost_base_jumprelu_threshold is required when --saeboost_base_infer_sparse_method=jumprelu."
        )
    if (
        args.saeboost_residual_infer_sparse_method == "jumprelu"
        and args.saeboost_residual_jumprelu_threshold is None
    ):
        raise ValueError(
            "--saeboost_residual_jumprelu_threshold is required when --saeboost_residual_infer_sparse_method=jumprelu."
        )
    if hasattr(args, "adv_bootstrap_repeats") and args.adv_bootstrap_repeats is not None:
        if args.adv_bootstrap_repeats < 0:
            raise ValueError("--adv_bootstrap_repeats must be non-negative.")
    if hasattr(args, "adv_bootstrap_ci") and args.adv_bootstrap_ci is not None:
        if args.adv_bootstrap_ci <= 0 or args.adv_bootstrap_ci >= 100:
            raise ValueError("--adv_bootstrap_ci must be in (0, 100).")
    if hasattr(args, "adv_ood_token_budget") and args.adv_ood_token_budget is not None:
        if args.adv_ood_token_budget <= 0:
            raise ValueError("--adv_ood_token_budget must be positive.")
    if hasattr(args, "faith_rank_top_f") and args.faith_rank_top_f is not None:
        if args.faith_rank_top_f <= 0:
            raise ValueError("--faith_rank_top_f must be positive.")
    if hasattr(args, "gae_rank_energy") and args.gae_rank_energy is not None:
        if args.gae_rank_energy <= 0.0 or args.gae_rank_energy > 1.0:
            raise ValueError("--gae_rank_energy must be in (0, 1].")
    if hasattr(args, "gae_rank_min") and args.gae_rank_min is not None:
        if args.gae_rank_min <= 0:
            raise ValueError("--gae_rank_min must be positive.")
    if hasattr(args, "gae_rank_max") and args.gae_rank_max is not None:
        if args.gae_rank_max <= 0:
            raise ValueError("--gae_rank_max must be positive when provided.")
    if (
        hasattr(args, "gae_rank_max")
        and hasattr(args, "gae_rank_min")
        and args.gae_rank_max is not None
        and args.gae_rank_min is not None
        and args.gae_rank_max < args.gae_rank_min
    ):
        raise ValueError("--gae_rank_max must be >= --gae_rank_min.")
    if hasattr(args, "gae_dict_method") and args.gae_dict_method is not None:
        print(f"Setting Config.GAE_DICT_METHOD: {Config.GAE_DICT_METHOD} -> {args.gae_dict_method}")
        Config.GAE_DICT_METHOD = args.gae_dict_method

    ckpt_kind, ckpt_output_kind = _inspect_checkpoint_type(args.checkpoint)
    if ckpt_kind is not None and ckpt_kind != args.explainer:
        raise ValueError(
            f"Checkpoint type '{ckpt_kind}' does not match --explainer '{args.explainer}'. "
            "Please pass a matching checkpoint."
        )
    if ckpt_output_kind is not None:
        print(f"Checkpoint output_kind detected: {ckpt_output_kind}")

    if args.baselines and "saeboost" in args.baselines and args.saeboost_residual_checkpoint:
        resid_ckpt = Path(args.saeboost_residual_checkpoint)
        if resid_ckpt.exists():
            resid_kind, resid_output_kind = _inspect_checkpoint_type(args.saeboost_residual_checkpoint)
            if resid_kind is not None and resid_kind != args.explainer:
                raise ValueError(
                    f"SAEBoost residual checkpoint type '{resid_kind}' does not match --explainer '{args.explainer}'."
                )
            if args.explainer == "transcoder" and resid_output_kind is not None and resid_output_kind != "mlp_out":
                raise ValueError(
                    f"SAEBoost transcoder residual checkpoint must be output_kind='mlp_out', got '{resid_output_kind}'."
                )
        elif not args.saeboost_train_residual:
            raise FileNotFoundError(
                f"SAEBoost residual checkpoint not found: {args.saeboost_residual_checkpoint} "
                "(pass --saeboost_train_residual to train it automatically)."
            )

    faith_m_values = [int(x) for x in args.faith_m.split(",") if x.strip()] if args.faith_m else []
    if len(faith_m_values) == 0:
        faith_m_values = None

    if args.task == 'adv':
        run_adversarial_ood_experiment(
            model_name=args.model,
            explainer_type=args.explainer,
            ood_set=args.ood_set,
            explainer_checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            n_eval_samples=args.n_eval,
            force_recompute_activations=args.force_recompute,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project or 'GAE_Adversarial-OOD',
            wandb_name=args.wandb_name,
            baselines=args.baselines,
            ood_retrained_max_epochs=args.ood_retrained_max_epochs,
            ood_retrained_max_steps=args.ood_retrained_max_steps,
            ood_retrained_total_tokens=args.ood_retrained_total_tokens,
            ood_retrained_lr=args.ood_retrained_lr,
            ood_retrained_lambda_sparse=args.ood_retrained_lambda_sparse,
            ood_retrained_objective=args.ood_retrained_objective,
            ood_retrained_term_t=args.ood_retrained_term_t,
            ood_retrained_batch_size=args.ood_retrained_batch_size,
            ood_retrained_streaming=args.ood_retrained_streaming,
            ood_retrained_context_size=args.ood_retrained_context_size,
            ood_retrained_warm_start=args.ood_retrained_warm_start,
            ood_retrained_reinit_b_dec=args.ood_retrained_reinit_b_dec,
            ood_retrained_use_saelens=args.ood_retrained_use_saelens,
            ood_retrained_checkpoint_path=args.ood_retrained_checkpoint,
            ood_retrained_checkpoint_dir=args.ood_retrained_checkpoint_dir,
            ood_finetuned_total_tokens=args.ood_finetuned_total_tokens,
            ood_finetuned_lr=args.ood_finetuned_lr,
            ood_finetuned_lambda_sparse=args.ood_finetuned_lambda_sparse,
            ood_finetuned_batch_size=args.ood_finetuned_batch_size,
            ood_finetuned_context_size=args.ood_finetuned_context_size,
            ood_finetuned_loss_type=args.ood_finetuned_loss_type,
            ood_finetuned_checkpoint_path=args.ood_finetuned_checkpoint_path,
            ood_finetuned_checkpoint_dir=args.ood_finetuned_checkpoint_dir,
            term_checkpoint_path=args.term_checkpoint,
            faithfulsae_checkpoint_path=getattr(args, 'faithfulsae_checkpoint', None),
            gae_r=args.gae_r,
            gae_rank_mode=args.gae_rank_mode,
            gae_rank_energy=args.gae_rank_energy,
            gae_rank_delta_mode=args.gae_rank_delta_mode,
            gae_rank_min=args.gae_rank_min,
            gae_rank_max=args.gae_rank_max,
            gae_decoder_mode=args.gae_decoder_mode,
            gae_dict_space=args.gae_dict_space,
            gae_recon_lambda=args.gae_recon_lambda,
            saeboost_residual_checkpoint=args.saeboost_residual_checkpoint,
            saeboost_residual_dir=args.saeboost_residual_dir,
            saeboost_coef=args.saeboost_coef,
            saeboost_train_residual=args.saeboost_train_residual,
            saeboost_residual_max_epochs=args.saeboost_residual_max_epochs,
            saeboost_residual_max_steps=args.saeboost_residual_max_steps,
            saeboost_residual_total_tokens=args.saeboost_residual_total_tokens,
            saeboost_residual_lr=args.saeboost_residual_lr,
            saeboost_residual_lambda_sparse=args.saeboost_residual_lambda_sparse,
            saeboost_residual_objective=args.saeboost_residual_objective,
            saeboost_residual_term_t=args.saeboost_residual_term_t,
            saeboost_residual_batch_size=args.saeboost_residual_batch_size,
            saeboost_residual_context_size=args.saeboost_residual_context_size,
            saeboost_residual_dict_size=args.saeboost_residual_dict_size,
            saeboost_residual_reinit_b_dec=args.saeboost_residual_reinit_b_dec,
            saeboost_residual_sparse_method=args.saeboost_residual_sparse_method,
            saeboost_residual_batch_top_k=args.saeboost_residual_batch_top_k,
            saeboost_residual_batch_top_k_start=args.saeboost_residual_batch_top_k_start,
            saeboost_residual_batch_top_k_warmup_fraction=args.saeboost_residual_batch_top_k_warmup_fraction,
            saeboost_base_infer_sparse_method=args.saeboost_base_infer_sparse_method,
            saeboost_base_infer_batch_top_k=args.saeboost_base_infer_batch_top_k,
            saeboost_base_jumprelu_threshold=args.saeboost_base_jumprelu_threshold,
            saeboost_residual_infer_sparse_method=args.saeboost_residual_infer_sparse_method,
            saeboost_residual_infer_batch_top_k=args.saeboost_residual_infer_batch_top_k,
            saeboost_residual_jumprelu_threshold=args.saeboost_residual_jumprelu_threshold,
            target_mode=args.target_mode,
            compute_ft=args.compute_ft,
            faith_m_values=faith_m_values,
            faith_random_repeats=args.faith_random_repeats,
            faith_m_star=args.faith_m_star,
            min_len_ood_override=args.min_len_ood,
            min_len_eval_override=args.min_len_eval,
            adv_ood_token_budget=args.adv_ood_token_budget,
            adv_bootstrap_repeats=args.adv_bootstrap_repeats,
            adv_bootstrap_ci=args.adv_bootstrap_ci,
        )
    elif args.task == 'domain':
        run_domain_ood_experiment(
            model_name=args.model,
            explainer_type=args.explainer,
            ood_set=args.ood_set,
            explainer_checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            n_eval_samples=args.n_eval,
            force_recompute_activations=args.force_recompute,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project or 'GAE_Domain-OOD',
            wandb_name=args.wandb_name,
            baselines=args.baselines,
            domain_ood_jsonl_dir=args.domain_ood_jsonl_dir,
            domain_ood_use_jsonl=not args.no_domain_ood_jsonl,
            domain_ood_jsonl_streaming=not args.no_domain_ood_jsonl_streaming,
            domain_ood_force_provided_target=args.domain_ood_force_provided_target,
            ood_retrained_max_epochs=args.ood_retrained_max_epochs,
            ood_retrained_max_steps=args.ood_retrained_max_steps,
            ood_retrained_total_tokens=args.ood_retrained_total_tokens,
            ood_retrained_lr=args.ood_retrained_lr,
            ood_retrained_lambda_sparse=args.ood_retrained_lambda_sparse,
            ood_retrained_objective=args.ood_retrained_objective,
            ood_retrained_term_t=args.ood_retrained_term_t,
            ood_retrained_batch_size=args.ood_retrained_batch_size,
            ood_retrained_streaming=args.ood_retrained_streaming,
            ood_retrained_context_size=args.ood_retrained_context_size,
            ood_retrained_warm_start=args.ood_retrained_warm_start,
            ood_retrained_reinit_b_dec=args.ood_retrained_reinit_b_dec,
            ood_retrained_use_saelens=args.ood_retrained_use_saelens,
            ood_retrained_checkpoint_path=args.ood_retrained_checkpoint,
            ood_retrained_checkpoint_dir=args.ood_retrained_checkpoint_dir,
            ood_finetuned_total_tokens=args.ood_finetuned_total_tokens,
            ood_finetuned_lr=args.ood_finetuned_lr,
            ood_finetuned_lambda_sparse=args.ood_finetuned_lambda_sparse,
            ood_finetuned_batch_size=args.ood_finetuned_batch_size,
            ood_finetuned_context_size=args.ood_finetuned_context_size,
            ood_finetuned_loss_type=args.ood_finetuned_loss_type,
            ood_finetuned_checkpoint_path=args.ood_finetuned_checkpoint_path,
            ood_finetuned_checkpoint_dir=args.ood_finetuned_checkpoint_dir,
            term_checkpoint_path=args.term_checkpoint,
            faithfulsae_checkpoint_path=getattr(args, 'faithfulsae_checkpoint', None),
            gae_r=args.gae_r,
            gae_rank_mode=args.gae_rank_mode,
            gae_rank_energy=args.gae_rank_energy,
            gae_rank_delta_mode=args.gae_rank_delta_mode,
            gae_rank_min=args.gae_rank_min,
            gae_rank_max=args.gae_rank_max,
            gae_decoder_mode=args.gae_decoder_mode,
            gae_dict_space=args.gae_dict_space,
            gae_recon_lambda=args.gae_recon_lambda,
            saeboost_residual_checkpoint=args.saeboost_residual_checkpoint,
            saeboost_residual_dir=args.saeboost_residual_dir,
            saeboost_coef=args.saeboost_coef,
            saeboost_train_residual=args.saeboost_train_residual,
            saeboost_residual_max_epochs=args.saeboost_residual_max_epochs,
            saeboost_residual_max_steps=args.saeboost_residual_max_steps,
            saeboost_residual_total_tokens=args.saeboost_residual_total_tokens,
            saeboost_residual_lr=args.saeboost_residual_lr,
            saeboost_residual_lambda_sparse=args.saeboost_residual_lambda_sparse,
            saeboost_residual_objective=args.saeboost_residual_objective,
            saeboost_residual_term_t=args.saeboost_residual_term_t,
            saeboost_residual_batch_size=args.saeboost_residual_batch_size,
            saeboost_residual_context_size=args.saeboost_residual_context_size,
            saeboost_residual_dict_size=args.saeboost_residual_dict_size,
            saeboost_residual_reinit_b_dec=args.saeboost_residual_reinit_b_dec,
            saeboost_residual_sparse_method=args.saeboost_residual_sparse_method,
            saeboost_residual_batch_top_k=args.saeboost_residual_batch_top_k,
            saeboost_residual_batch_top_k_start=args.saeboost_residual_batch_top_k_start,
            saeboost_residual_batch_top_k_warmup_fraction=args.saeboost_residual_batch_top_k_warmup_fraction,
            saeboost_base_infer_sparse_method=args.saeboost_base_infer_sparse_method,
            saeboost_base_infer_batch_top_k=args.saeboost_base_infer_batch_top_k,
            saeboost_base_jumprelu_threshold=args.saeboost_base_jumprelu_threshold,
            saeboost_residual_infer_sparse_method=args.saeboost_residual_infer_sparse_method,
            saeboost_residual_infer_batch_top_k=args.saeboost_residual_infer_batch_top_k,
            saeboost_residual_jumprelu_threshold=args.saeboost_residual_jumprelu_threshold,
            target_mode=args.target_mode,
            compute_ft=args.compute_ft,
            faith_m_values=faith_m_values,
            faith_random_repeats=args.faith_random_repeats,
            faith_m_star=args.faith_m_star,
        )
    else:
        run_timeshift_ood_experiment(
            model_name=args.model,
            explainer_type=args.explainer,
            ood_set=args.ood_set,
            explainer_checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            n_eval_samples=args.n_eval,
            force_recompute_activations=args.force_recompute,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project or 'GAE_TimeShift-OOD',
            wandb_name=args.wandb_name,
            baselines=args.baselines,
            ood_retrained_max_epochs=args.ood_retrained_max_epochs,
            ood_retrained_max_steps=args.ood_retrained_max_steps,
            ood_retrained_total_tokens=args.ood_retrained_total_tokens,
            ood_retrained_lr=args.ood_retrained_lr,
            ood_retrained_lambda_sparse=args.ood_retrained_lambda_sparse,
            ood_retrained_objective=args.ood_retrained_objective,
            ood_retrained_term_t=args.ood_retrained_term_t,
            ood_retrained_batch_size=args.ood_retrained_batch_size,
            ood_retrained_streaming=args.ood_retrained_streaming,
            ood_retrained_context_size=args.ood_retrained_context_size,
            ood_retrained_warm_start=args.ood_retrained_warm_start,
            ood_retrained_reinit_b_dec=args.ood_retrained_reinit_b_dec,
            ood_retrained_use_saelens=args.ood_retrained_use_saelens,
            ood_retrained_checkpoint_path=args.ood_retrained_checkpoint,
            ood_retrained_checkpoint_dir=args.ood_retrained_checkpoint_dir,
            ood_finetuned_total_tokens=args.ood_finetuned_total_tokens,
            ood_finetuned_lr=args.ood_finetuned_lr,
            ood_finetuned_lambda_sparse=args.ood_finetuned_lambda_sparse,
            ood_finetuned_batch_size=args.ood_finetuned_batch_size,
            ood_finetuned_context_size=args.ood_finetuned_context_size,
            ood_finetuned_loss_type=args.ood_finetuned_loss_type,
            ood_finetuned_checkpoint_path=args.ood_finetuned_checkpoint_path,
            ood_finetuned_checkpoint_dir=args.ood_finetuned_checkpoint_dir,
            term_checkpoint_path=args.term_checkpoint,
            faithfulsae_checkpoint_path=getattr(args, 'faithfulsae_checkpoint', None),
            gae_r=args.gae_r,
            gae_rank_mode=args.gae_rank_mode,
            gae_rank_energy=args.gae_rank_energy,
            gae_rank_delta_mode=args.gae_rank_delta_mode,
            gae_rank_min=args.gae_rank_min,
            gae_rank_max=args.gae_rank_max,
            gae_decoder_mode=args.gae_decoder_mode,
            gae_dict_space=args.gae_dict_space,
            gae_recon_lambda=args.gae_recon_lambda,
            saeboost_residual_checkpoint=args.saeboost_residual_checkpoint,
            saeboost_residual_dir=args.saeboost_residual_dir,
            saeboost_coef=args.saeboost_coef,
            saeboost_train_residual=args.saeboost_train_residual,
            saeboost_residual_max_epochs=args.saeboost_residual_max_epochs,
            saeboost_residual_max_steps=args.saeboost_residual_max_steps,
            saeboost_residual_total_tokens=args.saeboost_residual_total_tokens,
            saeboost_residual_lr=args.saeboost_residual_lr,
            saeboost_residual_lambda_sparse=args.saeboost_residual_lambda_sparse,
            saeboost_residual_objective=args.saeboost_residual_objective,
            saeboost_residual_term_t=args.saeboost_residual_term_t,
            saeboost_residual_batch_size=args.saeboost_residual_batch_size,
            saeboost_residual_context_size=args.saeboost_residual_context_size,
            saeboost_residual_dict_size=args.saeboost_residual_dict_size,
            saeboost_residual_reinit_b_dec=args.saeboost_residual_reinit_b_dec,
            saeboost_residual_sparse_method=args.saeboost_residual_sparse_method,
            saeboost_residual_batch_top_k=args.saeboost_residual_batch_top_k,
            saeboost_residual_batch_top_k_start=args.saeboost_residual_batch_top_k_start,
            saeboost_residual_batch_top_k_warmup_fraction=args.saeboost_residual_batch_top_k_warmup_fraction,
            saeboost_base_infer_sparse_method=args.saeboost_base_infer_sparse_method,
            saeboost_base_infer_batch_top_k=args.saeboost_base_infer_batch_top_k,
            saeboost_base_jumprelu_threshold=args.saeboost_base_jumprelu_threshold,
            saeboost_residual_infer_sparse_method=args.saeboost_residual_infer_sparse_method,
            saeboost_residual_infer_batch_top_k=args.saeboost_residual_infer_batch_top_k,
            saeboost_residual_jumprelu_threshold=args.saeboost_residual_jumprelu_threshold,
            target_mode=args.target_mode,
            compute_ft=args.compute_ft,
            faith_m_values=faith_m_values,
            faith_random_repeats=args.faith_random_repeats,
            faith_m_star=args.faith_m_star,
            faith_empty_mode=args.faith_empty_mode,
            faith_rank_mode=args.faith_rank_mode,
            faith_rank_top_f=args.faith_rank_top_f,
            faith_hook_site_mode=args.faith_hook_site_mode,
            timeshift_span_mode=args.timeshift_span_mode,
        )


if __name__ == '__main__':
    main()

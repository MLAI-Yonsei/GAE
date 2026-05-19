"""Consolidated utility module — merges former activation_cache, activation_utils, model_logits_utils, model_utils, text_utils, training_objectives."""

from __future__ import annotations
from config import Config
from pathlib import Path
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from typing import Any, Dict, Optional
from typing import List, Tuple, Dict
from typing import Optional, Tuple
import numpy as np
import random
import re
import string
import torch

# === activation_cache ===

def get_activation_cache_path(model_name, layer, data_type, n_samples=None, token_budget=None):
    """
    Get cache file path for activations.
    
    Args:
        model_name: Model identifier ('gpt2', 'pythia-410m', etc.)
        layer: Layer index
        data_type: 'id', 'ood_scientific', 'ood_code', etc.
        n_samples: Number of samples (optional, for filename)
        token_budget: Token budget (optional, for filename)
    
    Returns:
        cache_path: Path to cache file
    """
    if n_samples is not None and token_budget is not None:
        raise ValueError("Specify only one of n_samples or token_budget for cache path.")
    cache_dir = Path(Config.DATA_DIR) / "activations_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_name}_layer{layer}_{data_type}"
    if token_budget is not None:
        filename += f"_t{token_budget}"
    elif n_samples is not None:
        filename += f"_n{n_samples}"
    filename += ".pt"
    
    return cache_dir / filename


def find_activation_cache(model_name, layer, data_type, n_samples=None, token_budget=None):
    """
    Find activation cache file, checking multiple possible locations.
    
    Args:
        model_name: Model identifier
        layer: Layer index
        data_type: Data type identifier
        n_samples: Expected number of samples
        token_budget: Expected token budget
    
    Returns:
        cache_path: Path to cache file if found, None otherwise
    """
    if n_samples is not None and token_budget is not None:
        raise ValueError("Specify only one of n_samples or token_budget for cache lookup.")
    # Try current DATA_DIR location
    cache_path = get_activation_cache_path(model_name, layer, data_type, n_samples, token_budget)
    if cache_path.exists():
        return cache_path
    
    # Try without n_samples in filename
    if n_samples is not None:
        cache_path_no_n = get_activation_cache_path(model_name, layer, data_type, None)
        if cache_path_no_n.exists():
            return cache_path_no_n
    if token_budget is not None:
        cache_path_no_t = get_activation_cache_path(model_name, layer, data_type, None)
        if cache_path_no_t.exists():
            return cache_path_no_t
    

    
    # Also try relative 'data' from current working directory
    cwd_cache_dir = Path('data') / "activations_cache"
    if cwd_cache_dir.exists() and cwd_cache_dir.is_absolute() or (Path.cwd() / cwd_cache_dir).exists():
        actual_cache_dir = Path.cwd() / cwd_cache_dir if not cwd_cache_dir.is_absolute() else cwd_cache_dir
        filename = f"{model_name}_layer{layer}_{data_type}"
        if token_budget is not None:
            filename += f"_t{token_budget}"
        elif n_samples is not None:
            filename += f"_n{n_samples}"
        filename += ".pt"
        cwd_path = actual_cache_dir / filename
        if cwd_path.exists():
            print(f"Found cache in current working directory location: {cwd_path}")
            return cwd_path
        
        # Try without n_samples
        if n_samples is not None or token_budget is not None:
            cwd_path_no_n = actual_cache_dir / f"{model_name}_layer{layer}_{data_type}.pt"
            if cwd_path_no_n.exists():
                print(f"Found cache in current working directory location: {cwd_path_no_n}")
                return cwd_path_no_n
    
    return None


def _normalize_metadata_value(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_metadata_value(v) for k, v in value.items()}
    return value


def _metadata_mismatches(
    saved_metadata: Dict[str, Any],
    required_metadata: Dict[str, Any],
):
    mismatches = []
    for key, expected_value in required_metadata.items():
        expected_value = _normalize_metadata_value(expected_value)
        if key not in saved_metadata:
            mismatches.append(f"{key}: missing (expected {expected_value!r})")
            continue
        actual_value = _normalize_metadata_value(saved_metadata.get(key))
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: saved={actual_value!r}, expected={expected_value!r}"
            )
    return mismatches


def save_activations(
    activations,
    logits,
    model_name,
    layer,
    data_type,
    n_samples=None,
    token_budget=None,
    metadata_extra: Optional[Dict[str, Any]] = None,
):
    """
    Save activations and logits to disk.
    
    Args:
        activations: Activations tensor, shape [N, d_model]
        logits: Logits tensor, shape [N, vocab_size]
        model_name: Model identifier
        layer: Layer index
        data_type: Data type identifier
        n_samples: Number of samples (for filename)
        token_budget: Token budget (for filename)
    """
    cache_path = get_activation_cache_path(model_name, layer, data_type, n_samples, token_budget)
    
    cache_metadata = {
        'cache_version': 2,
        'model_name': model_name,
        'layer': int(layer),
        'data_type': data_type,
        'n_samples': len(activations) if n_samples is None else n_samples,
        'token_budget': token_budget,
    }
    if metadata_extra:
        cache_metadata.update(_normalize_metadata_value(metadata_extra))

    torch.save({
        'activations': activations,
        'logits': logits,
        'model_name': model_name,
        'layer': layer,
        'data_type': data_type,
        'n_samples': len(activations) if n_samples is None else n_samples,
        'token_budget': token_budget,
        'cache_metadata': cache_metadata,
        'config': {
            'max_len': Config.MAX_LEN,
            'min_len': Config.MIN_LEN,
            'seed': Config.SEED
        }
    }, cache_path)
    
    print(f"Saved activations to {cache_path}")
    print(f"  Shape: activations {activations.shape}, logits {logits.shape}")
    if metadata_extra:
        print(f"  Cache metadata: {cache_metadata}")


def load_activations(
    model_name,
    layer,
    data_type,
    n_samples=None,
    token_budget=None,
    check_config=True,
    required_metadata: Optional[Dict[str, Any]] = None,
):
    """
    Load activations and logits from disk.
    
    Args:
        model_name: Model identifier
        layer: Layer index
        data_type: Data type identifier
        n_samples: Expected number of samples (for validation)
        token_budget: Expected token budget (for validation)
        check_config: Whether to check config matches
    
    Returns:
        activations: Activations tensor
        logits: Logits tensor
        metadata: Dictionary with metadata
    """
    # Try to find cache file in multiple locations
    cache_path = find_activation_cache(model_name, layer, data_type, n_samples, token_budget)
    
    if cache_path is None:
        # List available cache files for debugging
        current_cache_dir = Path(Config.DATA_DIR) / "activations_cache"
        cwd_cache_dir = Path.cwd() / "data" / "activations_cache"
        
        print(f"Cache file not found. Searched locations:")
        print(f"  Current DATA_DIR: {current_cache_dir}")
        print(f"  Current working dir: {cwd_cache_dir}")
        
        # Check current location
        if current_cache_dir.exists():
            print(f"Available cache files in {current_cache_dir}:")
            for f in sorted(current_cache_dir.glob(f"{model_name}_layer{layer}_{data_type}*")):
                print(f"  - {f.name}")

        
        # Check cwd location
        if cwd_cache_dir.exists():
            print(f"Available cache files in {cwd_cache_dir}:")
            for f in sorted(cwd_cache_dir.glob(f"{model_name}_layer{layer}_{data_type}*")):
                print(f"  - {f.name}")
        
        return None, None, None
    
    print(f"Found cache file: {cache_path}")
    
    print(f"Loading activations from {cache_path}")
    data = torch.load(cache_path, map_location='cpu')
    
    activations = data['activations']
    logits = data['logits']
    cached_n_samples = len(activations)
    cache_metadata = data.get('cache_metadata', {})
    if not cache_metadata:
        cache_metadata = {
            'cache_version': 1,
            'model_name': data.get('model_name', model_name),
            'layer': data.get('layer', layer),
            'data_type': data.get('data_type', data_type),
            'n_samples': data.get('n_samples', cached_n_samples),
            'token_budget': data.get('token_budget', None),
        }
    metadata = {
        'model_name': data.get('model_name', model_name),
        'layer': data.get('layer', layer),
        'data_type': data.get('data_type', data_type),
        'n_samples': data.get('n_samples', cached_n_samples),
        'cached_n_samples': cached_n_samples,
        'token_budget': data.get('token_budget', None),
        'cache_path': str(cache_path),
        'cache_metadata': cache_metadata,
    }
    
    # Print cached sample count
    print(f"  Cached samples: {cached_n_samples}")
    
    # Validate
    if n_samples is not None and cached_n_samples != n_samples:
        print(f"Warning: Expected {n_samples} samples, got {cached_n_samples}")

    if required_metadata:
        mismatches = _metadata_mismatches(cache_metadata, required_metadata)
        if mismatches:
            print("Warning: Activation cache metadata mismatch detected.")
            for mismatch in mismatches:
                print(f"  {mismatch}")
            print("  Rejecting cache and forcing regeneration.")
            return None, None, None
        print("  Cache metadata validated.")
    
    if check_config:
        saved_config = data.get('config', {})
        current_config = {
            'max_len': Config.MAX_LEN,
            'min_len': Config.MIN_LEN,
            'seed': Config.SEED
        }
        
        mismatches = []
        for key in ['max_len', 'min_len', 'seed']:
            if key in saved_config and saved_config[key] != current_config[key]:
                mismatches.append(f"{key}: saved={saved_config[key]}, current={current_config[key]}")
        
        if mismatches:
            print(f"Warning: Config mismatches detected:")
            for mismatch in mismatches:
                print(f"  {mismatch}")
            print("  Consider regenerating activations with current config.")
    
    print(f"  Loaded: activations {activations.shape}, logits {logits.shape}")
    
    return activations, logits, metadata


# === activation_utils ===

def extract_residual_stream_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    position: int = -1,
    hook_name: str = "resid_post",
    stop_at_layer: int = None,
):
    """
    Extract residual stream activations at a specific layer and position.
    
    Args:
        model: HookedTransformer instance
        tokens: Input tokens, shape [batch_size, seq_len] or [seq_len]
        layer: Layer index (0-indexed)
        position: Position index (-1 for last token)
        hook_name: Hook name (default: "resid_post" for post-attention residual)
    
    Returns:
        activations: Residual stream activations, shape [batch_size, d_model]
    """
    model.eval()
    
    # Ensure tokens is 2D: [batch_size, seq_len]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)  # [seq_len] -> [1, seq_len]
    
    # Convert position to actual index
    if position < 0:
        position = tokens.shape[1] + position
    
    activations = None
    
    def activation_hook(value, hook):
        nonlocal activations
        # Extract activations at the specified position
        # value shape: [batch_size, seq_len, d_model]
        activations = value[:, position, :].clone()
        return value
    
    # Allow caller to pass a fully-qualified blocks.{layer}.{hook_name} or
    # a dotted hook like "ln2.hook_normalized" (already inside a sub-module).
    # If hook_name contains a dot, treat it as a full sub-path.
    if hook_name.startswith("blocks."):
        hook_name_full = hook_name
    elif "." in hook_name:
        hook_name_full = f"blocks.{layer}.{hook_name}"
    else:
        if not hook_name.startswith("hook_"):
            hook_name = f"hook_{hook_name}"
        hook_name_full = f"blocks.{layer}.{hook_name}"
    
    with torch.no_grad():
        run_kwargs = {}
        if stop_at_layer is not None:
            run_kwargs["stop_at_layer"] = stop_at_layer
        model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name_full, activation_hook)],
            **run_kwargs,
        )

    return activations


def extract_hook_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    position: int = -1,
    hook_name: str = "resid_post",
):
    """
    Extract activations at a specific hook point and position.
    """
    model.eval()

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    if position < 0:
        position = tokens.shape[1] + position

    activations = None

    def activation_hook(value, hook):
        nonlocal activations
        activations = value[:, position, :].clone()
        return value

    if hook_name.startswith("blocks."):
        hook_name_full = hook_name
    else:
        if "." not in hook_name and not hook_name.startswith("hook_"):
            hook_name = f"hook_{hook_name}"
        hook_name_full = f"blocks.{layer}.{hook_name}"

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name_full, activation_hook)])

    return activations


def extract_mlp_in_out_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    position: int = -1,
    hook_in: str = "ln2.hook_normalized",
    hook_out: str = "hook_mlp_out",
):
    """
    Extract MLP input/output activations at a specific layer/position.
    """
    model.eval()

    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    if position < 0:
        position = tokens.shape[1] + position

    act_in = None
    act_out = None

    def hook_in_fn(value, hook):
        nonlocal act_in
        act_in = value[:, position, :].clone()
        return value

    def hook_out_fn(value, hook):
        nonlocal act_out
        act_out = value[:, position, :].clone()
        return value

    if hook_in.startswith("blocks."):
        hook_in_full = hook_in
    else:
        if "." not in hook_in and not hook_in.startswith("hook_"):
            hook_in = f"hook_{hook_in}"
        hook_in_full = f"blocks.{layer}.{hook_in}"

    if hook_out.startswith("blocks."):
        hook_out_full = hook_out
    else:
        if "." not in hook_out and not hook_out.startswith("hook_"):
            hook_out = f"hook_{hook_out}"
        hook_out_full = f"blocks.{layer}.{hook_out}"

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_in_full, hook_in_fn), (hook_out_full, hook_out_fn)],
        )

    return act_in, act_out


def extract_logits(model: HookedTransformer, tokens: torch.Tensor, position: int = -1):
    """
    Extract logits at a specific position.
    
    Args:
        model: HookedTransformer instance
        tokens: Input tokens, shape [batch_size, seq_len]
        position: Position index (-1 for last token)
    
    Returns:
        logits: Logits at position, shape [batch_size, vocab_size]
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(tokens)  # [batch_size, seq_len, vocab_size]
    
    if position < 0:
        position = tokens.shape[1] + position
    
    return logits[:, position, :]


def collect_activations_batch(
    model: HookedTransformer,
    token_batches,
    layer: int,
    position: int = -1,
    device=None
):
    """
    Collect activations from multiple batches.
    
    Args:
        model: HookedTransformer instance
        token_batches: List of token tensors or a single tensor
        layer: Layer index
        position: Position index
        device: Device to use
    
    Returns:
        activations: Concatenated activations, shape [N, d_model]
    """
    if device is None:
        device = Config.device
    
    model.to(device)
    
    all_activations = []
    
    # Handle both list and single tensor
    if isinstance(token_batches, torch.Tensor):
        token_batches = [token_batches]
    
    with torch.no_grad():
        for tokens in token_batches:
            if tokens.device != device:
                tokens = tokens.to(device)
            
            # Ensure tokens is 2D: [batch_size, seq_len]
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)  # [seq_len] -> [1, seq_len]
            
            activations = extract_residual_stream_activations(
                model, tokens, layer, position
            )
            all_activations.append(activations.cpu())
    
    return torch.cat(all_activations, dim=0)


def collect_hook_activations_batch(
    model: HookedTransformer,
    token_batches,
    layer: int,
    position: int = -1,
    hook_name: str = "resid_post",
    device=None,
):
    """
    Collect activations from a specific hook point over multiple batches.
    """
    if device is None:
        device = Config.device

    model.to(device)
    all_activations = []

    if isinstance(token_batches, torch.Tensor):
        token_batches = [token_batches]

    with torch.no_grad():
        for tokens in token_batches:
            if tokens.device != device:
                tokens = tokens.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)

            activations = extract_hook_activations(
                model, tokens, layer, position=position, hook_name=hook_name
            )
            all_activations.append(activations.cpu())

    return torch.cat(all_activations, dim=0)


def collect_mlp_in_out_activations_batch(
    model: HookedTransformer,
    token_batches,
    layer: int,
    position: int = -1,
    hook_in: str = "ln2.hook_normalized",
    hook_out: str = "hook_mlp_out",
    device=None,
):
    """
    Collect MLP input/output activations from multiple batches.
    Returns (inputs, outputs) both shaped [N, d_model].
    """
    if device is None:
        device = Config.device

    model.to(device)
    all_in = []
    all_out = []

    if isinstance(token_batches, torch.Tensor):
        token_batches = [token_batches]

    with torch.no_grad():
        for tokens in token_batches:
            if tokens.device != device:
                tokens = tokens.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)

            h_in, h_out = extract_mlp_in_out_activations(
                model,
                tokens,
                layer,
                position=position,
                hook_in=hook_in,
                hook_out=hook_out,
            )
            all_in.append(h_in.cpu())
            all_out.append(h_out.cpu())

    return torch.cat(all_in, dim=0), torch.cat(all_out, dim=0)


# === model_logits_utils ===

MODEL_NAME_MAP = {
    "gpt2": "gpt2",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}


def load_hooked_model(
    model_name: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[HookedTransformer, AutoTokenizer]:
    """
    Load a TransformerLens HookedTransformer and its tokenizer.
    """
    tl_name = MODEL_NAME_MAP.get(model_name, model_name)
    model = HookedTransformer.from_pretrained(tl_name, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(tl_name)
    return model, tokenizer


def validate_logits_shape(logits: torch.Tensor, expected_vocab_size: int) -> None:
    """
    Ensure logits come from the model LM head, not an internal activation.
    """
    if logits.ndim != 3:
        raise RuntimeError(
            f"Invalid logits tensor: expected 3D [B,L,V], got shape {tuple(logits.shape)}"
        )
    if logits.shape[-1] != expected_vocab_size:
        raise RuntimeError(
            "Invalid logits tensor: last dim != vocab_size. "
            "You are likely reading an internal activation (d_model) instead of final logits. "
            f"Got last dim {logits.shape[-1]}, expected vocab_size {expected_vocab_size}."
        )


def get_true_logits_from_model(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run the model and return TRUE logits of shape [B, L, V].
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None:
        # HookedTransformer does not consistently use attention_mask; ignore if unsupported.
        try:
            logits = model(input_ids, attention_mask=attention_mask)
        except TypeError:
            logits = model(input_ids)
    else:
        logits = model(input_ids)

    validate_logits_shape(logits, expected_vocab_size=model.cfg.d_vocab)
    return logits


def get_target_logit(
    logits: torch.Tensor,
    position: int,
    target_id: int,
) -> torch.Tensor:
    """
    Extract target logit at a specific position. Returns [B] tensor.
    """
    if position < 0:
        position = logits.shape[1] + position
    return logits[:, position, target_id]


def _resolve_hook_name(layer: int, hook_site: str) -> str:
    if hook_site.startswith("blocks."):
        return hook_site
    # Dotted hook paths are already explicit module paths under blocks.{layer}
    # (e.g. "ln2.hook_normalized"). Some checkpoints may store
    # "hook_ln2.hook_normalized"; normalize the first segment.
    if "." in hook_site:
        first, rest = hook_site.split(".", 1)
        if first.startswith("hook_"):
            first = first[len("hook_") :]
        return f"blocks.{layer}.{first}.{rest}"
    if hook_site.startswith("hook_"):
        return f"blocks.{layer}.{hook_site}"
    return f"blocks.{layer}.hook_{hook_site}"


def forward_with_override_logits(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    layer: int,
    position: Optional[int],
    hook_site: str,
    h_override: torch.Tensor,
) -> torch.Tensor:
    """
    Inject overridden activations at a hook point and return full logits [B, L, V].

    Shape-polymorphic dispatch via h_override.dim():
      * dim() == 1 : [d_model]        -> unsqueeze to [1, d_model], single-position override
      * dim() == 2 : [B, d_model]     -> single-position override at `position`
      * dim() == 3 : [B, L, d_model]  -> full-sequence override at ALL positions; `position` is IGNORED
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    full_seq = (h_override.dim() == 3)

    if full_seq:
        if h_override.shape[0] != input_ids.shape[0]:
            raise RuntimeError(
                f"Full-seq override batch mismatch: h_override batch={h_override.shape[0]} "
                f"input_ids batch={input_ids.shape[0]}"
            )
        if h_override.shape[1] != input_ids.shape[1]:
            raise RuntimeError(
                f"Full-seq override length mismatch: h_override L={h_override.shape[1]} "
                f"input_ids L={input_ids.shape[1]}"
            )
        position_resolved = None
    else:
        if h_override.dim() == 1:
            h_override = h_override.unsqueeze(0)
        if position is None:
            raise RuntimeError("position must be provided for single-position (2-D h_override) override")
        position_resolved = input_ids.shape[1] + position if position < 0 else position

    hook_name = _resolve_hook_name(layer, hook_site)

    def _override_hook(act, hook):
        # act: [B, L, d_model]
        if act.dim() != 3:
            raise RuntimeError(f"Expected hook activation [B,L,d_model], got {tuple(act.shape)}")
        if h_override.shape[0] != act.shape[0]:
            raise RuntimeError(
                f"Batch mismatch for override: h_override batch={h_override.shape[0]} act batch={act.shape[0]}"
            )
        act = act.clone()
        if full_seq:
            act[:, :, :] = h_override
        else:
            act[:, position_resolved, :] = h_override
        return act

    logits = model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, _override_hook)])
    validate_logits_shape(logits, expected_vocab_size=model.cfg.d_vocab)
    return logits


def forward_with_override_and_get_target_logit(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
    hook_site: str,
    h_override: torch.Tensor,
    target_id: int,
    return_logits: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Inject overridden activations at (layer, position) and return target logit.

    Args:
        model: HookedTransformer
        input_ids: [B, L] or [L]
        layer: layer index
        position: token position (supports negative indexing)
        hook_site: hook name like "resid_post" or "mlp_out" (or full hookpoint)
        h_override: [B, d_model] (or [d_model] for single example)
        target_id: target token id
        return_logits: if True, also return full logits [B, L, V]
    """
    logits = forward_with_override_logits(
        model=model,
        input_ids=input_ids,
        layer=layer,
        position=position,
        hook_site=hook_site,
        h_override=h_override,
    )
    target_logit = get_target_logit(logits, position=position, target_id=target_id)
    if return_logits:
        return target_logit, logits
    return target_logit, None


# === model_utils ===

def set_seed(seed=None):
    """Set random seed for reproducibility."""
    if seed is None:
        seed = Config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if Config.DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_model(model_name, device=None):
    """
    Load a TransformerLens model.
    
    Args:
        model_name: Model identifier (e.g., 'gpt2', 'pythia-410m', 'pythia-1.4b')
        device: Device to load model on (default: Config.device)
    
    Returns:
        model: HookedTransformer instance
        tokenizer: Tokenizer instance
    """
    if device is None:
        device = Config.device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    
    if model_name not in Config.MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(Config.MODELS.keys())}")
    
    model_path = Config.MODELS[model_name]
    
    print(f"Loading model: {model_path}")
    model = HookedTransformer.from_pretrained(
        model_path,
        device=device,
        dtype=torch.bfloat16 if Config.USE_BF16 else (torch.float16 if Config.USE_FP16 else torch.float32)
    )
    model.eval()
    
    # Disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    tokenizer = model.tokenizer
    
    print(f"Model loaded: {model_name}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hidden dim: {model.cfg.d_model}")
    print(f"  Vocab size: {model.cfg.d_vocab}")
    print(f"  Device: {device} ({'GPU' if device.type == 'cuda' else 'CPU'})")
    
    return model, tokenizer


def get_layer(model_name):
    """Get the layer index for activation extraction."""
    return Config.LAYERS.get(model_name, None)


# === text_utils ===

STOP_TOKENS = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were","be","been","it","this","that"
}

OVERRIDE_CONNECTORS = ["however", "notwithstanding", "except when", "provided that", "unless"]

EDGAR_LEADINS = ["expected to be", "is likely to be", "will be", "may become"]
PATENT_LEADINS = ["is configured to", "thereby", "such that the output", "the controller then"]

EDGAR_BAIT = ["may be adversely affected", "could materially affect", "risk", "uncertainty", "adversely"]
PATENT_BAIT = ["at least one", "triggers an alert", "in some embodiments", "configured to", "threshold"]

EDGAR_KEYWORDS = ["risk", "uncertainty", "adverse", "material", "affect", "liquidity", "regulatory", "market", "interest", "currency", "credit"]
PATENT_KEYWORDS = ["sensor", "signal", "threshold", "detect", "anomaly", "alert", "controller", "module"]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_segments(text: str, min_chars: int = 300, max_chars: int = 2000) -> List[str]:
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    segments = []
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        if len(b) < min_chars:
            continue
        if len(b) > max_chars:
            # sentence pack
            sentences = re.split(r"(?<=[.!?])\s+", b)
            cur = []
            cur_len = 0
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if cur_len + len(s) + 1 > max_chars and cur:
                    segments.append(" ".join(cur).strip())
                    cur = [s]
                    cur_len = len(s)
                else:
                    cur.append(s)
                    cur_len += len(s) + 1
            if cur:
                segments.append(" ".join(cur).strip())
        else:
            segments.append(b)
    return segments


def _extract_clause(segment: str, keywords: List[str], fallback: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", segment)
    random.shuffle(sents)
    for s in sents:
        ls = s.lower()
        if any(k in ls for k in keywords):
            return s.strip()
    return fallback


def enforce_2hop(text: str, dataset: str) -> str:
    if "if" in text and ("only if" in text or "unless" in text):
        return text
    if dataset == "edgar":
        return text + " if demand decreases only if supply constraints persist"
    return text + " if the signal exceeds the threshold unless the calibration module is active"


def build_edgar_template(segment: str) -> Tuple[str, Dict[str, str]]:
    X = _extract_clause(
        segment,
        EDGAR_KEYWORDS,
        "market conditions deteriorate and liquidity tightens"
    )
    X = enforce_2hop(X, "edgar")
    Y = _extract_clause(
        segment,
        ["hedge", "diversif", "mitigat", "insurance", "offset", "reduce"],
        "we have mitigation measures and diversified exposures"
    )
    override = "however"
    bait = "may be adversely affected"
    lead_in = "expected to be"
    template = f"Our business may be adversely affected by {X}; {override}, {Y}, and therefore the impact is {lead_in}"
    return template, {"override": override, "bait": bait, "lead_in": lead_in}


def build_patent_template(segment: str) -> Tuple[str, Dict[str, str]]:
    X = _extract_clause(
        segment,
        PATENT_KEYWORDS,
        "sensor detects an anomaly"
    )
    X = enforce_2hop(X, "patent")
    Y = _extract_clause(
        segment,
        ["calibrat", "diagnostic", "transient", "filter"],
        "the calibration module is active"
    )
    override = "except when"
    bait = "triggers an alert"
    lead_in = "is configured to"
    template = f"The system triggers an alert when at least one {X}, {override} {Y}, in which case the controller {lead_in}"
    return template, {"override": override, "bait": bait, "lead_in": lead_in}


def pick_leadin_variant(dataset: str) -> str:
    return random.choice(EDGAR_LEADINS if dataset == "edgar" else PATENT_LEADINS)


def pick_override_variant() -> str:
    return random.choice(OVERRIDE_CONNECTORS)


def is_bad_target(token_str: str) -> bool:
    s = token_str.strip().lower()
    if s == "":
        return True
    if all(ch in string.punctuation for ch in s):
        return True
    if s in STOP_TOKENS:
        return True
    return False


def last80_contains(text: str, phrases: List[str]) -> bool:
    lt = text.lower()
    return all(p in lt for p in phrases)


def has_2hop(text: str) -> bool:
    t = text.lower()
    return ("if" in t and "only if" in t) or ("if" in t and "unless" in t)


# === training_objectives ===

def erm_loss(per_sample_losses):
    """
    ERM: Expected Risk Minimization
    L = E[L]
    
    Args:
        per_sample_losses: Tensor of per-sample losses [batch_size]
    
    Returns:
        Mean loss value
    """
    return per_sample_losses.mean()


def term_loss(per_sample_losses, t):
    """
    TERM: Tilted Empirical Risk Minimization
    L = (1/t) * log(E[exp(t*L)])
    
    For t > 0: emphasizes high-loss samples (larger t = stronger emphasis)
    As t -> 0: approaches ERM (mean loss)
    As t -> inf: approaches max loss
    
    Args:
        per_sample_losses: Tensor of per-sample losses [batch_size]
        t: Tilt parameter (t > 0)
    
    Returns:
        TERM loss value (always >= mean loss by Jensen's inequality)
    """
    if t <= 0:
        raise ValueError("TERM tilt parameter t must be positive")
    
    mean_loss = per_sample_losses.mean()
    
    # For very small t, use Taylor expansion to avoid numerical instability
    loss_scale = mean_loss.item() if mean_loss.item() > 0 else 1.0
    if t * loss_scale < 1e-2:
        # Use second-order Taylor expansion for numerical stability
        mean_loss_sq = (per_sample_losses ** 2).mean()
        var_loss = mean_loss_sq - mean_loss ** 2
        return mean_loss + (t / 2.0) * var_loss
    
    # For larger t, use stable log-sum-exp trick
    max_loss = per_sample_losses.max()
    shifted_losses = per_sample_losses - max_loss
    
    exp_arg = t * shifted_losses
    exp_arg_clamped = torch.clamp(exp_arg, min=-50.0, max=0.0)
    exp_terms = torch.exp(exp_arg_clamped)
    
    mean_exp = exp_terms.mean()
    log_mean_exp = torch.log(mean_exp + 1e-10) + max_loss
    
    term_loss_value = (1.0 / t) * log_mean_exp
    
    # Ensure TERM >= ERM (by Jensen's inequality)
    return torch.maximum(term_loss_value, mean_loss)


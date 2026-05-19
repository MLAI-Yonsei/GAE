"""
Utilities for caching activations to disk.
"""
import torch
from pathlib import Path
from typing import Any, Dict, Optional
from config import Config


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

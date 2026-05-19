"""
Utilities for extracting activations from TransformerLens models.
"""
import torch
from transformer_lens import HookedTransformer
from config import Config


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

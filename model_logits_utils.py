"""
Utilities to get TRUE model logits and to run forward passes with activation overrides.
All target scores MUST be taken from final model logits (last dim == vocab_size).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer


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

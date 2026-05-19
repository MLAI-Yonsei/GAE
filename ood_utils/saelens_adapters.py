"""
SAELens adapters for OOD evaluation/training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from saeboost import SAEBoostExplainerAdapter, SAEBoostTranscoderAdapter


def _strip_blocks_prefix(hook_point: str):
    if hook_point is None:
        return None
    if hook_point.startswith("blocks."):
        parts = hook_point.split(".", 2)
        if len(parts) == 3:
            return parts[2]
    return hook_point


class _SaeLensEncoder(nn.Module):
    def __init__(self, saelens):
        super().__init__()
        self.saelens = saelens
        self.activation_fn = getattr(saelens.cfg, "activation_fn", "relu") if hasattr(saelens, "cfg") else "relu"
        self.topk_k = getattr(saelens.cfg, "topk_k", None) if hasattr(saelens, "cfg") else None

    def forward(self, h):
        h = h.to(self.saelens.dtype)
        sae_in = h - self.saelens.b_dec
        hidden_pre = sae_in @ self.saelens.W_enc + self.saelens.b_enc
        if self.activation_fn == "topk" and self.topk_k is not None:
            topk_values, topk_indices = torch.topk(hidden_pre, self.topk_k, dim=-1)
            result = torch.zeros_like(hidden_pre)
            result.scatter_(-1, topk_indices, F.relu(topk_values))
            return result
        return F.relu(hidden_pre)


class _SaeLensDecoder(nn.Module):
    def __init__(self, saelens):
        super().__init__()
        self.saelens = saelens

    def forward(self, z):
        if getattr(self.saelens.cfg, "is_transcoder", False):
            return z @ self.saelens.W_dec + self.saelens.b_dec_out
        return z @ self.saelens.W_dec + self.saelens.b_dec


class SaeLensSparseAutoencoderAdapter(nn.Module):
    """
    Adapter for SAELens SparseAutoencoder checkpoints to match the OOD explainer API.
    """
    def __init__(self, saelens_model):
        super().__init__()
        self.saelens = saelens_model
        self.k = saelens_model.d_sae
        self.encoder = _SaeLensEncoder(saelens_model)
        self.decoder = _SaeLensDecoder(saelens_model)
        self.output_kind = "sae"
        self.input_hook_point = _strip_blocks_prefix(getattr(saelens_model.cfg, "hook_point", None))
        self.output_hook_point = _strip_blocks_prefix(getattr(saelens_model.cfg, "out_hook_point", None))

    def forward(self, h, return_z=False):
        sae_out, feature_acts, *_ = self.saelens(h)
        if return_z:
            return sae_out, feature_acts
        return sae_out


class SaeLensTranscoderAdapter(nn.Module):
    """
    Adapter for SAELens Transcoder checkpoints to match the OOD explainer API.
    """
    def __init__(self, saelens_model):
        super().__init__()
        self.saelens = saelens_model
        self.k = saelens_model.d_sae
        self.encoder = _SaeLensEncoder(saelens_model)
        self.decoder = _SaeLensDecoder(saelens_model)
        self.output_kind = "mlp_out"
        self.input_hook_point = _strip_blocks_prefix(getattr(saelens_model.cfg, "hook_point", None))
        self.output_hook_point = _strip_blocks_prefix(getattr(saelens_model.cfg, "out_hook_point", None))

    @property
    def W_tilde(self):
        return self.saelens.W_dec.T

    @property
    def b_tilde(self):
        return self.saelens.b_dec_out

    def forward(self, h, return_z=False):
        out, feature_acts, *_ = self.saelens(h)
        if return_z:
            return out, feature_acts
        return out


EXPLAINER_TYPES = (
    SaeLensTranscoderAdapter,
    SaeLensSparseAutoencoderAdapter,
    SAEBoostTranscoderAdapter,
    SAEBoostExplainerAdapter,
)

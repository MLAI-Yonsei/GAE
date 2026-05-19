"""
SAEBoost adapters that combine a fixed base explainer with a residual explainer.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _extract_w_dec(explainer: nn.Module) -> torch.Tensor:
    """
    Return decoder weights in shape [k, d].
    """
    if hasattr(explainer, "saelens") and hasattr(explainer.saelens, "W_dec"):
        return explainer.saelens.W_dec
    if hasattr(explainer, "W_dec"):
        return explainer.W_dec
    dec = getattr(explainer, "decoder", None)
    if dec is not None and hasattr(dec, "weight"):
        return dec.weight
    raise AttributeError(
        "Could not find decoder weights (expected one of saelens.W_dec, W_dec, decoder.weight)."
    )


def _extract_cfg_field(explainer: nn.Module, key: str, default=None):
    cfg = getattr(getattr(explainer, "saelens", None), "cfg", None)
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def _saeboost_batchtopk_feature_acts(feature_acts: torch.Tensor, top_k_per_sample: int) -> torch.Tensor:
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


def _resolve_sparse_spec(
    explainer: nn.Module,
    sparse_method: Optional[str] = None,
    batch_top_k: Optional[int] = None,
    jumprelu_threshold: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    method = str(sparse_method).lower() if sparse_method is not None else "auto"
    if method == "auto":
        inferred = _extract_cfg_field(explainer, "saeboost_inference_sparse_method", None)
        if inferred is None:
            inferred = _extract_cfg_field(explainer, "saeboost_sparse_method", None)
        method = str(inferred).lower() if inferred is not None else "none"
    if method in ("l1", "relu"):
        method = "none"
    if method not in ("none", "batchtopk", "jumprelu"):
        raise ValueError(f"Unsupported sparse inference method: {method}")

    k_value = batch_top_k
    if k_value is None:
        k_value = _extract_cfg_field(explainer, "saeboost_batch_top_k", None)
    if k_value is None:
        k_value = _extract_cfg_field(explainer, "top_k", None)
    if k_value is not None:
        k_value = int(k_value)
        if k_value <= 0:
            k_value = None

    threshold = jumprelu_threshold
    if threshold is None:
        threshold = _extract_cfg_field(explainer, "saeboost_jumprelu_threshold", None)
    if threshold is not None:
        threshold = float(threshold)

    return {"method": method, "batch_top_k": k_value, "jumprelu_threshold": threshold}


def _forward_with_sparse_mode(
    explainer: nn.Module,
    x: torch.Tensor,
    sparse_spec: Dict[str, Optional[float]],
    return_z: bool,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    method = str(sparse_spec.get("method", "none")).lower()
    if method == "none":
        return explainer(x, return_z=return_z)

    encoder = getattr(explainer, "encoder", None)
    decoder = getattr(explainer, "decoder", None)
    if encoder is None or decoder is None:
        return explainer(x, return_z=return_z)

    dense_feature_acts = encoder(x)
    if method == "batchtopk":
        k_value = sparse_spec.get("batch_top_k", None)
        if k_value is None:
            raise ValueError("batchtopk inference requires batch_top_k.")
        feature_acts = _saeboost_batchtopk_feature_acts(dense_feature_acts, int(k_value))
    elif method == "jumprelu":
        threshold = sparse_spec.get("jumprelu_threshold", None)
        if threshold is None:
            raise ValueError("jumprelu inference requires jumprelu_threshold.")
        thr = torch.as_tensor(float(threshold), device=dense_feature_acts.device, dtype=dense_feature_acts.dtype)
        feature_acts = dense_feature_acts * (dense_feature_acts > thr).to(dense_feature_acts.dtype)
    else:
        raise ValueError(f"Unsupported sparse inference method: {method}")

    recon = decoder(feature_acts)
    if return_z:
        return recon, feature_acts
    return recon


class SAEBoostExplainerAdapter(nn.Module):
    """
    Base SAE + residual SAE adapter.
    """

    def __init__(
        self,
        base_explainer: nn.Module,
        resid_explainer: nn.Module,
        resid_coef: float = 1.0,
        base_sparse_method: Optional[str] = None,
        base_batch_top_k: Optional[int] = None,
        base_jumprelu_threshold: Optional[float] = None,
        resid_sparse_method: Optional[str] = None,
        resid_batch_top_k: Optional[int] = None,
        resid_jumprelu_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.base_explainer = base_explainer
        self.resid_explainer = resid_explainer
        self.resid_coef = float(resid_coef)
        self._disable_residual = self.resid_coef == 0.0

        self.output_kind = getattr(base_explainer, "output_kind", "sae")
        self.input_hook_point = getattr(base_explainer, "input_hook_point", None)
        self.output_hook_point = getattr(base_explainer, "output_hook_point", None)

        self.k_base = int(getattr(base_explainer, "k", _extract_w_dec(base_explainer).shape[0]))
        self.k_resid = int(getattr(resid_explainer, "k", _extract_w_dec(resid_explainer).shape[0]))
        self.k = self.k_base if self._disable_residual else self.k_base + self.k_resid

        self.base_sparse_spec = _resolve_sparse_spec(
            base_explainer,
            sparse_method=base_sparse_method,
            batch_top_k=base_batch_top_k,
            jumprelu_threshold=base_jumprelu_threshold,
        )
        self.resid_sparse_spec = _resolve_sparse_spec(
            resid_explainer,
            sparse_method=resid_sparse_method,
            batch_top_k=resid_batch_top_k,
            jumprelu_threshold=resid_jumprelu_threshold,
        )

    @property
    def W_dec(self) -> torch.Tensor:
        base_w = _extract_w_dec(self.base_explainer)
        if self._disable_residual:
            return base_w
        resid_w = _extract_w_dec(self.resid_explainer)
        return torch.cat([base_w, self.resid_coef * resid_w], dim=0)

    def forward(self, h: torch.Tensor, return_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self._disable_residual:
            return _forward_with_sparse_mode(
                self.base_explainer,
                h,
                sparse_spec=self.base_sparse_spec,
                return_z=return_z,
            )
        h_base, z_base = _forward_with_sparse_mode(
            self.base_explainer,
            h,
            sparse_spec=self.base_sparse_spec,
            return_z=True,
        )
        h_resid, z_resid = _forward_with_sparse_mode(
            self.resid_explainer,
            h,
            sparse_spec=self.resid_sparse_spec,
            return_z=True,
        )
        h_boost = h_base + self.resid_coef * h_resid
        if not return_z:
            return h_boost
        z_boost = torch.cat([z_base, self.resid_coef * z_resid], dim=-1)
        return h_boost, z_boost

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        if self._disable_residual:
            return self.base_explainer.decoder(z)
        z_base = z[..., : self.k_base]
        z_resid = z[..., self.k_base :]
        h_base = self.base_explainer.decoder(z_base)
        h_resid = self.resid_explainer.decoder(z_resid)
        return h_base + self.resid_coef * h_resid


class SAEBoostTranscoderAdapter(nn.Module):
    """
    Base MLP-out transcoder + residual MLP-out transcoder adapter.
    """

    def __init__(
        self,
        base_transcoder: nn.Module,
        resid_transcoder: nn.Module,
        resid_coef: float = 1.0,
        base_sparse_method: Optional[str] = None,
        base_batch_top_k: Optional[int] = None,
        base_jumprelu_threshold: Optional[float] = None,
        resid_sparse_method: Optional[str] = None,
        resid_batch_top_k: Optional[int] = None,
        resid_jumprelu_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.base_transcoder = base_transcoder
        self.resid_transcoder = resid_transcoder
        self.resid_coef = float(resid_coef)
        self._disable_residual = self.resid_coef == 0.0

        self.output_kind = "mlp_out"
        self.input_hook_point = getattr(base_transcoder, "input_hook_point", None)
        self.output_hook_point = getattr(base_transcoder, "output_hook_point", None)

        self.k_base = int(getattr(base_transcoder, "k", _extract_w_dec(base_transcoder).shape[0]))
        self.k_resid = int(getattr(resid_transcoder, "k", _extract_w_dec(resid_transcoder).shape[0]))
        self.k = self.k_base if self._disable_residual else self.k_base + self.k_resid

        self.base_sparse_spec = _resolve_sparse_spec(
            base_transcoder,
            sparse_method=base_sparse_method,
            batch_top_k=base_batch_top_k,
            jumprelu_threshold=base_jumprelu_threshold,
        )
        self.resid_sparse_spec = _resolve_sparse_spec(
            resid_transcoder,
            sparse_method=resid_sparse_method,
            batch_top_k=resid_batch_top_k,
            jumprelu_threshold=resid_jumprelu_threshold,
        )

    @property
    def W_dec(self) -> torch.Tensor:
        base_w = _extract_w_dec(self.base_transcoder)
        if self._disable_residual:
            return base_w
        resid_w = _extract_w_dec(self.resid_transcoder)
        return torch.cat([base_w, self.resid_coef * resid_w], dim=0)

    def forward(self, x_in: torch.Tensor, return_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self._disable_residual:
            return _forward_with_sparse_mode(
                self.base_transcoder,
                x_in,
                sparse_spec=self.base_sparse_spec,
                return_z=return_z,
            )
        y_base, z_base = _forward_with_sparse_mode(
            self.base_transcoder,
            x_in,
            sparse_spec=self.base_sparse_spec,
            return_z=True,
        )
        y_resid, z_resid = _forward_with_sparse_mode(
            self.resid_transcoder,
            x_in,
            sparse_spec=self.resid_sparse_spec,
            return_z=True,
        )
        y_boost = y_base + self.resid_coef * y_resid
        if not return_z:
            return y_boost
        z_boost = torch.cat([z_base, self.resid_coef * z_resid], dim=-1)
        return y_boost, z_boost

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        if self._disable_residual:
            return self.base_transcoder.decoder(z)
        z_base = z[..., : self.k_base]
        z_resid = z[..., self.k_base :]
        y_base = self.base_transcoder.decoder(z_base)
        y_resid = self.resid_transcoder.decoder(z_resid)
        return y_base + self.resid_coef * y_resid

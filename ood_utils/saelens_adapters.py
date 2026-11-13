"""Frozen SAE/Transcoder checkpoint loading without training or SAEBoost code."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn


class CheckpointConfig(SimpleNamespace):
    """Data-only representation of the original checkpoint configuration."""


def load_payload(path):
    allowed=[(CheckpointConfig,'sae_training.config.LanguageModelSAERunnerConfig')]
    with torch.serialization.safe_globals(allowed):
        return torch.load(Path(path),map_location='cpu',weights_only=True)


class FrozenExplainer(nn.Module):
    def __init__(self,state,cfg,kind):
        super().__init__();self.kind=kind
        required=['W_enc','W_dec','b_enc','b_dec']+(['b_dec_out'] if kind=='transcoder' else [])
        for name in required:
            value=state.get(name)
            if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError('Missing or nonfinite checkpoint tensor: '+name)
            if value.dtype!=torch.float32:raise ValueError('This release requires FP32 explainer checkpoints')
            self.register_buffer(name,value.detach().contiguous().clone())
        din,width=self.W_enc.shape
        if self.b_dec.shape!=(din,) or self.b_enc.shape!=(width,) or self.W_dec.shape[0]!=width:
            raise ValueError('Checkpoint encoder/decoder shapes differ')
        if kind=='sae' and self.W_dec.shape[1]!=din:raise ValueError('SAE input/output dimensions differ')
        if kind=='transcoder' and self.b_dec_out.shape!=(self.W_dec.shape[1],):raise ValueError('Transcoder output bias differs')
        self.activation_fn=cfg.get('activation_fn','relu');self.topk_k=cfg.get('topk_k')
        if self.activation_fn not in ['relu','topk']:raise ValueError('Only ReLU and TopK checkpoints are supported')
        if self.activation_fn=='topk' and (not isinstance(self.topk_k,int) or not 0<self.topk_k<=width):
            raise ValueError('Checkpoint has invalid TopK')
        self.input_hook_point=cfg.get('hook_point') or 'ln2.hook_normalized'
        layer='.'.join(self.input_hook_point.split('.')[:2]) if self.input_hook_point.startswith('blocks.') else None
        self.output_hook_point=self.input_hook_point if kind=='sae' else cfg.get('out_hook_point') or (layer+'.hook_mlp_out' if layer else 'hook_mlp_out')
        self.output_kind='sae' if kind=='sae' else 'mlp_out'

    def encoder(self,x):
        pre=(x.to(self.W_enc.dtype)-self.b_dec)@self.W_enc+self.b_enc
        if self.activation_fn=='topk':
            values,indices=pre.topk(self.topk_k,dim=-1)
            return torch.zeros_like(pre).scatter(-1,indices,values.relu())
        return pre.relu()

    def decoder(self,z):return z@self.W_dec+(self.b_dec_out if self.kind=='transcoder' else self.b_dec)

    def forward(self,x,return_z=False):
        z=self.encoder(x);y=self.decoder(z)
        return (y,z) if return_z else y


def load_explainer(path,kind=None,device='cpu'):
    payload=load_payload(path)
    if not isinstance(payload,dict) or not isinstance(payload.get('state_dict'),dict):
        raise ValueError('Checkpoint must contain cfg and state_dict with W_enc/W_dec/b_enc/b_dec')
    cfg=payload.get('cfg',{});cfg=dict(cfg) if isinstance(cfg,dict) else vars(cfg)
    state=payload['state_dict'];actual='transcoder' if state.get('b_dec_out') is not None else 'sae'
    if kind is not None and kind!=actual:raise ValueError('Requested explainer kind differs from checkpoint')
    return FrozenExplainer(state,cfg,actual).to(device).eval().requires_grad_(False)


class DecoderAdapter(nn.Module):
    def __init__(self,base,weight,bias):
        super().__init__();self.base=base
        if weight.shape!=base.W_dec.T.shape or bias.shape!=(base.W_dec.shape[1],):raise ValueError('Adapted decoder shape differs')
        self.register_buffer('weight',weight.detach().T.contiguous().T)
        self.register_buffer('bias',bias.detach().contiguous())
        self.input_hook_point=base.input_hook_point;self.output_hook_point=base.output_hook_point
        self.output_kind=base.output_kind

    def encoder(self,x):return self.base.encoder(x)
    def decoder(self,z):return z@self.weight.T+self.bias
    def forward(self,x,return_z=False):
        z=self.encoder(x);y=self.decoder(z)
        return (y,z) if return_z else y

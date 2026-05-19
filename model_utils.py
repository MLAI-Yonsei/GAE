"""
Utilities for loading TransformerLens models.
"""
import torch
import random
import numpy as np
from transformer_lens import HookedTransformer
from config import Config


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

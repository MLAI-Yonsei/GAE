"""
Configuration for Domain-OOD and Noise-OOD experiments.
"""
import os

import torch

class Config:
    # Random seed for reproducibility
    SEED = 2026
    
    # Model configurations
    MODELS = {
        'gpt2': 'gpt2',
        'pythia-410m': 'EleutherAI/pythia-410m',
        'pythia-1.4b': 'EleutherAI/pythia-1.4b',
        'gemma-3-1b': 'google/gemma-3-1b-pt',
    }

    # Layer selection (per model)
    LAYERS = {
        'gpt2': 8,  # mid-layer for GPT-2 Small (12 layers total)
        'pythia-410m': 15,  # mid-layer for Pythia-410M (24 layers total)
        'pythia-1.4b': 15,  # mid-layer for Pythia-1.4B (24 layers total)
        'gemma-3-1b': 19,  # 3/4 depth for Gemma-3-1B (26 layers total)
    }
    
    # Data parameters
    MAX_LEN = 256  # Maximum sequence length
    MIN_LEN = 64   # Minimum sequence length after tokenization
    BATCH_SIZE = 64 # 1024 for gpt2, 256 for pythia-410m, 64 for pythia-1.4b  # Batch size for inference (reduced for larger models)
    
    # Explainer parameters
    DICT_SIZE_K = {
        'gpt2': 24576,
        'pythia-410m': 32768,
        'pythia-1.4b': 65536,
        'gemma-3-1b': 36864,  # 1152 * 32 (32x expansion)
    }
    OOD_SUBSPACE_RANK_R = 64  # OOD active subspace rank
    
    # Dataset sizes
    N_OOD_DOMAIN = 20000  # Number of OOD domain sequences
    N_OOD_NOISE = 20000  # Number of OOD noise sequences
    N_COV = int(os.environ.get("N_COV", "2000"))  # Number of activations for OOD covariance estimation
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def print_device_info():
        """Print device information."""
        print(f"Device: {Config.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("  Using CPU (GPU not available)")
    
    # Precision
    USE_BF16 = False  # Use bfloat16 for inference if stable
    USE_FP16 = False  # Use float16 for inference if stable

    # Decoder-only refit (used for *_refit metrics). Disabled to avoid OOM.
    ENABLE_DECODER_REFIT = False

    # GAE dictionary extraction.
    # method: "normal" (Z^T Z) | "row" (Z Z^T) | "auto" (row when N < K)
    GAE_DICT_METHOD = "normal"
    GAE_DICT_DEVICE = "cpu"  # "cpu" or "cuda"
    GAE_DICT_RIDGE = 1e-8

    # Diagnostics reconstruction (for geometric metrics).
    # method: "normal" (Z^T Z) | "row" (Z Z^T) | "auto" (row when B < K)
    DIAG_RECON_METHOD = "auto"

    # Faithfulness normalization: delta_max values below this threshold are treated
    # as unstable for normalized metrics (nAOPC / nSuff / nComp primary scores).
    FAITH_NORM_MIN_DELTA = float(os.environ.get("FAITH_NORM_MIN_DELTA", "0.1"))

    # Affine Constrained GAE (decoder prior + affine bias).
    GAE_AC_LAMBDA_GEOM = float(os.environ.get("GAE_AC_LAMBDA_GEOM", "1e-1"))
    GAE_AC_LAMBDA_ID = float(os.environ.get("GAE_AC_LAMBDA_ID", "5e-2"))
    GAE_AC_LAMBDA_GAE = float(os.environ.get("GAE_AC_LAMBDA_GAE", "2e-1"))
    GAE_AC_BATCH_SIZE = int(os.environ.get("GAE_AC_BATCH_SIZE", "128"))
    GAE_AC_FIT_SAMPLES = int(os.environ.get("GAE_AC_FIT_SAMPLES", "2048"))
    GAE_AC_SOLVER_DEVICE = os.environ.get("GAE_AC_SOLVER_DEVICE", "cpu")
    GAE_AC_MATCH_COLUMN_NORMS = os.environ.get("GAE_AC_MATCH_COLUMN_NORMS", "0") == "1"
    GAE_AC_DECODER_MIX_PRIOR = float(os.environ.get("GAE_AC_DECODER_MIX_PRIOR", "0.0"))
    GAE_AC_DECODER_MIX_ORIG = float(os.environ.get("GAE_AC_DECODER_MIX_ORIG", "0.0"))
    GAE_AC_BIAS_MIX_ORIG = float(os.environ.get("GAE_AC_BIAS_MIX_ORIG", "0.0"))
    GAE_AC_LAMBDA_RES = float(os.environ.get("GAE_AC_LAMBDA_RES", "0.1"))

    # SAE-specific GAE modes.
    GAE_SAE_ENCODER_MIX = float(os.environ.get("GAE_SAE_ENCODER_MIX", "1.0"))
    GAE_SAE_ENCODER_PRESERVE_ORTHOGONAL = os.environ.get("GAE_SAE_ENCODER_PRESERVE_ORTHOGONAL", "1") == "1"
    GAE_SAE_SEL_MODE = os.environ.get("GAE_SAE_SEL_MODE", "ood_active")
    GAE_AC_DIAG_LAM = float(os.environ.get("GAE_AC_DIAG_LAM", "1e-3"))
    GAE_AC_DIAG_CLAMP_MIN = float(os.environ.get("GAE_AC_DIAG_CLAMP_MIN", "0.5"))
    GAE_AC_DIAG_CLAMP_MAX = float(os.environ.get("GAE_AC_DIAG_CLAMP_MAX", "2.0"))

    # CE-sensitivity weighted decoder (Step 2 enhancement).
    GAE_CE_WEIGHTS = os.environ.get("GAE_CE_WEIGHTS", "0") == "1"
    GAE_CE_WEIGHTS_N = int(os.environ.get("GAE_CE_WEIGHTS_N", "512"))
    GAE_CE_WEIGHTS_EPS = float(os.environ.get("GAE_CE_WEIGHTS_EPS", "1e-8"))
    GAE_CE_WEIGHTS_ALPHA = float(os.environ.get("GAE_CE_WEIGHTS_ALPHA", "0.5"))

    # SC (support-preserving coefficient refit) hyperparameters.
    GAE_SUPPORT_REFIT_RIDGE = float(os.environ.get("GAE_SUPPORT_REFIT_RIDGE", "1e-3"))
    GAE_SUPPORT_REFIT_TOPK = int(os.environ.get("GAE_SUPPORT_REFIT_TOPK", "128"))

    # GAE correction strength: 0~1 = manual. Default 1.0 (full correction).
    GAE_CORRECTION_ALPHA = float(os.environ.get("GAE_CORRECTION_ALPHA", "1.0"))
    # GAE per-component weighting: weight_j = alpha * (1 - sigma_j)
    GAE_PER_COMPONENT = os.environ.get("GAE_PER_COMPONENT", "0") == "1"
    # GAE iterative refinement: number of iterations (1 = standard, 2+ = iterative)
    GAE_N_ITERS = int(os.environ.get("GAE_N_ITERS", "1"))

    # SAELens OOD-retrain buffer sizing (None = auto)
    SAELENS_STORE_BATCH_SIZE = None
    SAELENS_N_BATCHES_IN_BUFFER = None
    
    # Determinism flags
    DETERMINISTIC = True
    
    # Paths
    OUTPUT_DIR = 'results'
    DATA_DIR = os.environ.get('TC_DATA_DIR', os.environ.get('REPO_DATA', './data') + '/data')
    
    # HuggingFace and datasets cache directory
    HF_CACHE_DIR = os.environ.get('DATA_ROOT', './data') + '/hf_cache'
    DATASETS_CACHE_DIR = os.environ.get('DATA_ROOT', './data') + '/datasets_cache'

    @staticmethod
    def get_dict_size_k(model_name: str) -> int:
        key = model_name.lower()
        if key in Config.DICT_SIZE_K:
            return Config.DICT_SIZE_K[key]
        if model_name in Config.DICT_SIZE_K:
            return Config.DICT_SIZE_K[model_name]
        raise KeyError(f"DICT_SIZE_K has no entry for model '{model_name}'.")

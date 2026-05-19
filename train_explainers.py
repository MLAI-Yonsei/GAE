"""
Utilities for ID activation collection and SAELens explainer training.
"""
import importlib
import os
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_utils import load_model, set_seed, get_layer
from activation_utils import extract_mlp_in_out_activations
from activation_cache import load_activations, save_activations
from config import Config
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_training.train_sae_on_language_model import train_sae_on_language_model

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


def collect_id_activations(
    model,
    tokenizer,
    n_samples,
    layer,
    total_tokens=None,
    hook_in="ln2.hook_normalized",
    hook_out="hook_mlp_out",
    position=-1,
    device=None,
    use_cache=True,
    force_recompute=False,
    data_module=None,
):
    """
    Collect ID activations for training explainers.

    Args:
        model: HookedTransformer instance
        tokenizer: Tokenizer instance
        n_samples: Number of activation samples to collect
        total_tokens: Token budget to collect (streaming)
        hook_in: MLP input hook name (for mlp_in_out)
        hook_out: MLP output hook name (for mlp_in_out)
        position: Token position to extract (default: last token)
        layer: Layer index
        device: Device to use
        use_cache: If True, try to load from cache first
        force_recompute: If True, recompute even if cache exists
        data_module: Module with load_id_dataset/sample_and_tokenize_dataset

    Returns:
        activations: Collected input activations, shape [n_samples, d_model]
        targets: Collected targets (logits or MLP outputs), shape [n_samples, d_out]
        metadata: Dictionary with metadata including cached_n_samples
    """
    if device is None:
        device = Config.device

    # Determine model name
    model_name = None
    for name, path in Config.MODELS.items():
        if path in str(model.cfg.model_name) or name in str(model.cfg.model_name):
            model_name = name
            break
    if model_name is None:
        model_name = 'gpt2'  # Default

    cache_metadata = {
        "dataset_role": "id",
        "extraction_family": "mlp_in_out",
        "activation_kind": "hook_input",
        "target_kind": "hook_output",
        "input_hook_point": hook_in,
        "output_hook_point": hook_out,
        "position": int(position),
    }

    # Try to load from cache
    if use_cache and not force_recompute:
        print(f"Attempting to load cached activations for {model_name}, layer {layer}, n_samples={n_samples}")
        activations, targets, metadata = load_activations(
            model_name,
            layer,
            "id_mlp",
            n_samples,
            token_budget=total_tokens,
            check_config=True,
            required_metadata=cache_metadata,
        )

        if activations is not None:
            cached_n_samples = len(activations)
            # Use cached activations even if we have fewer samples than requested
            # Recomputing would likely yield the same number of samples due to dataset limitations
            if n_samples is None:
                print(f"Using cached ID activations ({cached_n_samples} samples cached)")
                return activations, targets, metadata
            if cached_n_samples >= n_samples:
                print(f"Using cached ID activations ({cached_n_samples} samples cached, using {n_samples})")
                return activations[:n_samples], targets[:n_samples], metadata
            print(f"Using cached ID activations ({cached_n_samples} samples cached, requested {n_samples})")
            print("Note: Dataset may not have enough valid samples. Recomputing would likely yield the same result.")
            # Return all cached samples (up to n_samples)
            return activations[:min(cached_n_samples, n_samples)], targets[:min(cached_n_samples, n_samples)], metadata
        else:
            print("No cached activations found. Will compute new activations.")

    # Collect activations (cache miss or force recompute)
    print("Collecting ID activations (this may take a while)...")
    model.to(device)
    model.eval()

    if data_module is None:
        data_module = importlib.import_module("ood_utils.data_utils")
        print("data_module not provided; defaulting to ood_utils.data_utils")

    dataset, text_field = data_module.load_id_dataset(model_name)

    # Sample and tokenize
    if total_tokens is not None:
        print(f"Sampling sequences from ID dataset until {total_tokens} tokens...")
    else:
        print(f"Sampling {n_samples} sequences from ID dataset...")
    tokenized = data_module.sample_and_tokenize_dataset(
        dataset, text_field, tokenizer, n_samples,
        total_tokens=total_tokens,
        max_length=Config.MAX_LEN, min_length=Config.MIN_LEN,
        seed=Config.SEED
    )

    collected_tokens = sum(int(t.shape[0]) for t in tokenized)
    print(f"Collected {len(tokenized)} valid sequences ({collected_tokens} tokens)")

    # Extract activations in batches
    all_activations = []
    all_targets = []
    batch_size = Config.BATCH_SIZE

    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized), batch_size), desc="Extracting activations"):
            batch_tokens = tokenized[i:i+batch_size]

            # Pad to same length
            max_len = max(t.shape[0] for t in batch_tokens)
            padded_tokens = []
            for t in batch_tokens:
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    padded = torch.cat([t, torch.zeros(pad_len, dtype=t.dtype)])
                else:
                    padded = t
                padded_tokens.append(padded)

            tokens_batch = torch.stack(padded_tokens).to(device)

            # Extract activations and targets
            activations, targets = extract_mlp_in_out_activations(
                model,
                tokens_batch,
                layer,
                position=position,
                hook_in=hook_in,
                hook_out=hook_out,
            )

            all_activations.append(activations.cpu())
            all_targets.append(targets.cpu())

            if n_samples is not None and len(all_activations) * batch_size >= n_samples:
                break

    if n_samples is None:
        activations = torch.cat(all_activations, dim=0)
        targets = torch.cat(all_targets, dim=0)
    else:
        activations = torch.cat(all_activations, dim=0)[:n_samples]
        targets = torch.cat(all_targets, dim=0)[:n_samples]

    # Create metadata
    metadata = {
        'model_name': model_name,
        'layer': layer,
        'data_type': 'id',
        'n_samples': len(activations) if n_samples is None else n_samples,
        'cached_n_samples': len(activations),
        'token_budget': total_tokens,
        'collected_tokens': collected_tokens,
        'hook_in': hook_in,
        'hook_out': hook_out,
    }

    # Save to cache
    if use_cache:
        save_activations(
            activations,
            targets,
            model_name,
            layer,
            "id_mlp",
            n_samples,
            token_budget=total_tokens,
            metadata_extra=cache_metadata,
        )
        print(f"Saved {len(activations)} activations to cache")

    return activations, targets, metadata


def main():
    """Main training script (SAELens-style)."""
    import argparse

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description='Train explainers with activation store buffers')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(Config.MODELS.keys()),
                       help='Model to use')
    parser.add_argument('--explainer', type=str, required=True,
                       choices=['transcoder', 'sae', 'both'],
                       help='Explainer type to train')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Dataset path (HF dataset name)')
    parser.add_argument('--dataset_split', type=str, default='train',
                       help='Dataset split (e.g. "train", "train[:5%%]")')
    parser.add_argument('--dataset_streaming', action='store_true',
                       help='Use streaming mode (default: download to local cache)')
    parser.add_argument('--context_size', type=int, default=128,
                       help='Context size for token batching')
    parser.add_argument('--total_tokens', type=int, default=60000000,
                       help='Total training tokens')
    parser.add_argument('--train_batch_size', type=int, default=4096,
                       help='Training batch size')
    parser.add_argument('--store_batch_size', type=int, default=32,
                       help='Activation store batch size')
    parser.add_argument('--n_batches_in_buffer', type=int, default=128,
                       help='Number of batches in activation buffer')
    parser.add_argument('--dict_size_k', type=int, default=None,
                       help='Dictionary size (k). Must be multiple of d_model.')
    parser.add_argument('--expansion_factor', type=int, default=32,
                       help='Expansion factor (d_sae = d_in * expansion_factor)')
    parser.add_argument('--lr', type=float, default=5e-2,
                       help='Learning rate')
    parser.add_argument('--l1_coefficient', type=float, default=1.4e-4,
                       help='L1 sparsity coefficient')
    parser.add_argument('--objective', type=str, default='ERM',
                       choices=['ERM', 'TERM'],
                       help='Training objective')
    parser.add_argument('--term_t', type=float, default=0.001,
                       help='TERM tilt parameter')
    parser.add_argument('--lr_scheduler_name', type=str, default='constantwithwarmup',
                       choices=['constant', 'constantwithwarmup', 'linearwarmupdecay', 'cosineannealing', 'cosineannealingwarmup'],
                       help='LR scheduler name')
    parser.add_argument('--lr_warm_up_steps', type=int, default=5000,
                       help='Warmup steps')
    parser.add_argument('--b_dec_init_method', type=str, default='mean',
                       choices=['geometric_median', 'mean', 'zeros'],
                       help='b_dec init method')
    parser.add_argument('--hook_point', type=str, default=None,
                       help='Hook point for inputs')
    parser.add_argument('--hook_point_layer', type=int, default=None,
                       help='Hook point layer index')
    parser.add_argument('--out_hook_point', type=str, default=None,
                       help='Output hook point for transcoders')
    parser.add_argument('--out_hook_point_layer', type=int, default=None,
                       help='Output hook point layer index')
    parser.add_argument('--use_cached_activations', action='store_true',
                       help='Use cached activations from disk')
    parser.add_argument('--cached_activations_path', type=str, default=None,
                       help='Path to cached activations (dir)')
    parser.add_argument('--activation_fn', type=str, default='relu',
                       choices=['relu', 'topk'],
                       help='Activation function: relu (vanilla L1 SAE) or topk (TopK SAE)')
    parser.add_argument('--topk_k', type=int, default=None,
                       help='Number of top-k features to keep (required when activation_fn=topk)')
    parser.add_argument('--topk_aux_coeff', type=float, default=1.0/32,
                       help='TopK auxiliary dead-feature loss coefficient (default: 1/32)')
    parser.add_argument('--use_ghost_grads', action='store_true',
                       help='Enable ghost gradients')
    parser.add_argument('--feature_sampling_method', type=str, default=None,
                       choices=[None, 'l2', 'anthropic'],
                       help='Feature resampling method')
    parser.add_argument('--feature_sampling_window', type=int, default=1000,
                       help='Feature sampling window')
    parser.add_argument('--resample_batches', type=int, default=32,
                       help='Resample batches')
    parser.add_argument('--dead_feature_window', type=int, default=5000,
                       help='Dead feature window')
    parser.add_argument('--dead_feature_threshold', type=float, default=1e-8,
                       help='Dead feature threshold')
    parser.add_argument('--n_checkpoints', type=int, default=0,
                       help='Number of checkpoints to save during training')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--device', type=str, default=None,
                       help='Device override (e.g., cuda)')
    parser.add_argument('--seed', type=int, default=Config.SEED,
                       help='Random seed')

    args = parser.parse_args()

    set_seed(args.seed)

    model_name = args.model
    model, _ = load_model(model_name)
    layer = get_layer(model_name)
    d_in = model.cfg.d_model
    # SAELens needs the HF model ID, not our short alias
    saelens_model_name = Config.MODELS.get(model_name, model_name)

    if args.dataset_path is None:
        if model_name == 'gpt2':
            dataset_path = "Skylion007/openwebtext"
        elif model_name.startswith('gemma'):
            dataset_path = os.environ.get("DATA_ROOT", "./data") + "/datasets_cache/redpajama_v2_sample.jsonl"
        else:
            dataset_path = "monology/pile-uncopyrighted"
    else:
        dataset_path = args.dataset_path

    if args.hook_point_layer is None:
        args.hook_point_layer = layer
    if args.out_hook_point_layer is None:
        args.out_hook_point_layer = layer

    if args.hook_point is None:
        args.hook_point = f"blocks.{args.hook_point_layer}.ln2.hook_normalized"
    if args.out_hook_point is None:
        args.out_hook_point = f"blocks.{args.out_hook_point_layer}.hook_mlp_out"

    if args.dict_size_k is None:
        args.dict_size_k = Config.get_dict_size_k(model_name)

    expansion_factor = args.expansion_factor
    if args.dict_size_k is not None:
        if args.dict_size_k % d_in != 0:
            raise ValueError("dict_size_k must be a multiple of d_model to match SAELens expansion_factor.")
        expansion_factor = args.dict_size_k // d_in
    if expansion_factor is None:
        expansion_factor = 32

    def _train_one(is_transcoder: bool):
        cfg = LanguageModelSAERunnerConfig(
            model_name=saelens_model_name,
            hook_point=args.hook_point,
            hook_point_layer=args.hook_point_layer,
            dataset_path=dataset_path,
            dataset_split=args.dataset_split,
            dataset_streaming=args.dataset_streaming,
            datasets_cache_dir=os.environ.get('HF_DATASETS_CACHE', None),
            is_dataset_tokenized=False,
            context_size=args.context_size,
            use_cached_activations=args.use_cached_activations,
            cached_activations_path=args.cached_activations_path,
            d_in=d_in,
            expansion_factor=expansion_factor,
            b_dec_init_method=args.b_dec_init_method,
            lr=args.lr,
            l1_coefficient=args.l1_coefficient,
            objective_type=args.objective,
            term_t=args.term_t,
            lr_scheduler_name=args.lr_scheduler_name,
            lr_warm_up_steps=args.lr_warm_up_steps,
            train_batch_size=args.train_batch_size,
            n_batches_in_buffer=args.n_batches_in_buffer,
            total_training_tokens=args.total_tokens,
            store_batch_size=args.store_batch_size,
            activation_fn=args.activation_fn,
            topk_k=args.topk_k,
            topk_aux_coeff=args.topk_aux_coeff,
            use_ghost_grads=args.use_ghost_grads,
            feature_sampling_method=args.feature_sampling_method,
            feature_sampling_window=args.feature_sampling_window,
            resample_batches=args.resample_batches,
            dead_feature_window=args.dead_feature_window,
            dead_feature_threshold=args.dead_feature_threshold,
            log_to_wandb=args.use_wandb,
            wandb_project=args.wandb_project or "mats_sae_training_language_model",
            n_checkpoints=args.n_checkpoints,
            checkpoint_path=args.checkpoint_path,
            device=args.device or str(Config.device),
            seed=args.seed,
            is_transcoder=is_transcoder,
            out_hook_point=args.out_hook_point if is_transcoder else None,
            out_hook_point_layer=args.out_hook_point_layer if is_transcoder else None,
            d_out=d_in if is_transcoder else None,
        )

        run_suffix = "transcoder" if is_transcoder else "sae"
        run_name = args.wandb_name or f"{cfg.run_name}-{run_suffix}"
        cfg.wandb_name = run_name
        cfg.checkpoint_path = os.path.join(args.checkpoint_path, run_name)

        wandb_run = None
        if args.use_wandb:
            if WANDB_AVAILABLE:
                wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    name=run_name,
                    config=cfg.__dict__,
                    reinit=True,
                )
            else:
                print("Warning: --use_wandb was set but wandb is not available. Skipping wandb logging.")

        loader = LMSparseAutoencoderSessionloader(cfg)
        model_local, sparse_autoencoder, activations_loader = loader.load_session()
        try:
            sparse_autoencoder = train_sae_on_language_model(
                model_local,
                sparse_autoencoder,
                activations_loader,
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
        finally:
            if wandb_run is not None:
                wandb.finish()

        out_path = os.path.join(cfg.checkpoint_path, f"final_{sparse_autoencoder.get_name()}.pt")
        sparse_autoencoder.save_model(out_path)

    if args.explainer in ("sae", "both"):
        _train_one(is_transcoder=False)
    if args.explainer in ("transcoder", "both"):
        _train_one(is_transcoder=True)


if __name__ == '__main__':
    main()

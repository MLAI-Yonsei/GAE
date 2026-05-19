"""
Evaluate explainer faithfulness on In-Distribution (ID) data.
Used to diagnose whether low OOD performance stems from checkpoint quality
or genuine distribution shift effects.

Usage:
    python eval_id_faithfulness.py \
        --model pythia-1.4b \
        --explainer transcoder \
        --checkpoint /path/to/checkpoint.pt \
        --n_eval 500 \
        --batch_size 64
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm

from config import Config
from utils import load_model, set_seed, get_layer
from run_experiment import load_explainer, evaluate_baseline_fixed
from ood_utils.data_utils import load_id_dataset, _extract_text
from ood_utils.evaluation import FAITH_HOOK_SITE_MODES

# Set HuggingFace cache directories
os.environ['HF_HOME'] = Config.HF_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = Config.HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = Config.DATASETS_CACHE_DIR


def sample_id_tokens(dataset, text_field, tokenizer, n_samples, max_length, min_length, seed=2026):
    """Sample and tokenize ID data."""
    import random
    rng = random.Random(seed)

    tokens_list = []
    is_streaming = hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__')

    if is_streaming:
        iterator = iter(dataset)
        attempts = 0
        max_attempts = n_samples * 10
        while len(tokens_list) < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                example = next(iterator)
            except StopIteration:
                break
            text = _extract_text(example, text_field)
            if not text or len(text.strip()) == 0:
                continue
            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length)
            tokens = tokens.squeeze(0)
            if len(tokens) >= min_length:
                tokens_list.append(tokens)
    else:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        for idx in indices:
            if len(tokens_list) >= n_samples:
                break
            example = dataset[idx]
            text = _extract_text(example, text_field)
            if not text or len(text.strip()) == 0:
                continue
            tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length)
            tokens = tokens.squeeze(0)
            if len(tokens) >= min_length:
                tokens_list.append(tokens)

    return tokens_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate explainer faithfulness on ID data")
    parser.add_argument("--model", type=str, required=True, choices=["gpt2", "pythia-410m", "pythia-1.4b"])
    parser.add_argument("--explainer", type=str, required=True, choices=["transcoder", "sae"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_eval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--target_mode", type=str, default="argmax")
    parser.add_argument("--faith_rank_mode", type=str, default="causal_topz")
    parser.add_argument("--faith_rank_top_f", type=int, default=64)
    parser.add_argument("--faith_m_star", type=int, default=32)
    parser.add_argument("--faith_empty_mode", type=str, default="zero_resid")
    parser.add_argument("--faith_hook_site_mode", type=str, default="auto")
    args = parser.parse_args()

    set_seed(Config.SEED)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    layer = get_layer(args.model)
    d_model = model.cfg.d_model
    vocab_size = model.cfg.d_vocab
    print(f"Model config: d_model={d_model}, vocab_size={vocab_size}, layer={layer}")

    # Load explainer
    print(f"\nLoading explainer: {args.checkpoint}")
    explainer = load_explainer(args.checkpoint, args.explainer, d_model, vocab_size)

    # Load ID dataset
    print(f"\nLoading ID dataset for {args.model}...")
    dataset, text_field = load_id_dataset(args.model)

    # Sample and tokenize
    print(f"Sampling {args.n_eval} ID sequences...")
    eval_tokens_list = sample_id_tokens(
        dataset, text_field, tokenizer,
        n_samples=args.n_eval,
        max_length=Config.MAX_LEN,
        min_length=Config.MIN_LEN,
        seed=Config.SEED,
    )
    print(f"Collected {len(eval_tokens_list)} ID evaluation sequences")

    if len(eval_tokens_list) == 0:
        raise RuntimeError("No valid ID sequences found. Check dataset and length constraints.")

    # Evaluate faithfulness on ID data
    print(f"\nEvaluating ID faithfulness ({len(eval_tokens_list)} samples)...")
    results = evaluate_baseline_fixed(
        explainer=explainer,
        model=model,
        tokens_list=eval_tokens_list,
        layer=layer,
        model_type=args.explainer,
        device=Config.device,
        target_mode=args.target_mode,
        faith_m_star=args.faith_m_star,
        faith_empty_mode=args.faith_empty_mode,
        faith_rank_mode=args.faith_rank_mode,
        faith_rank_top_f=args.faith_rank_top_f,
        faith_hook_site_mode=args.faith_hook_site_mode,
    )

    # Print summary
    print("\n" + "=" * 80)
    print(f"ID Faithfulness Results — {args.model} / {args.explainer}")
    print("=" * 80)

    key_metrics = [
        ("mean_auc", "Logit AUC"),
        ("n_aopc_primary_mean", "nAOPC (primary)"),
        ("comp_m_mean", "Comp"),
        ("suff_m_mean", "Suff"),
        ("hidden_auc_mean", "Hidden AUC"),
    ]
    for key, label in key_metrics:
        val = results.get(key, None)
        if val is not None:
            std_key = key.replace("_mean", "_std").replace("mean_auc", "std_auc")
            std = results.get(std_key, 0)
            print(f"  {label:20s}: {val:.4f} ± {std:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"id_faithfulness_{args.model}_{args.explainer}.json"
    )
    results["model"] = args.model
    results["explainer"] = args.explainer
    results["checkpoint"] = args.checkpoint
    results["n_eval"] = len(eval_tokens_list)
    results["setting"] = "in_distribution"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

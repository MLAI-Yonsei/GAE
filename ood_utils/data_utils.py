"""Shared dataset utilities for OOD experiments."""
import logging
import warnings
from datasets import load_dataset, IterableDataset
try:
    from datasets import DatasetDict, IterableDatasetDict
    HAS_DATASET_DICTS = True
except Exception:
    HAS_DATASET_DICTS = False
from transformers import AutoTokenizer
import numpy as np
import torch
from config import Config

# Suppress datasets library warnings about deprecated scripts
logging.getLogger("datasets").setLevel(logging.ERROR)


def load_id_dataset(model_name):
    """
    Load ID dataset based on model family.

    Args:
        model_name: Model identifier ('gpt2', 'pythia-410m', 'pythia-1.4b')

    Returns:
        dataset: HuggingFace dataset
        text_field: Name of the text field in the dataset
    """
    if model_name == 'gpt2':
        dataset = load_dataset("openwebtext", split="train")
        text_field = "text"
    elif model_name.startswith('pythia'):
        # The Pile dataset - using monology/pile-uncopyrighted (publicly available)
        # The original EleutherAI/pile no longer supports dataset scripts
        # Using streaming mode for efficient loading of large dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        text_field = "text"
    elif model_name.startswith('gemma'):
        # RedPajama v2 for Gemma models (following "Transcoders Beat SAEs" protocol)
        redpajama_path = os.environ.get("DATA_ROOT", "./data") + "/datasets_cache/redpajama_v2_sample.jsonl"
        dataset = load_dataset("json", data_files=redpajama_path, split="train")
        text_field = "text"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return dataset, text_field


def _extract_text(example, text_field):
    """
    Extract text from a dataset example.

    text_field can be:
      - str key
      - tuple/list of nested keys
      - callable(example) -> text
    """
    if callable(text_field):
        return text_field(example)
    if isinstance(text_field, (tuple, list)):
        cur = example
        for key in text_field:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur
    if isinstance(text_field, str):
        return example.get(text_field) if isinstance(example, dict) else None
    return None


def _pick_text_field(dataset, preferred_fields):
    """Pick the first available text field from preferred_fields."""
    if not preferred_fields:
        return None
    # Try dataset.features when available
    try:
        if hasattr(dataset, "features") and dataset.features is not None:
            available = set(dataset.features.keys())
            for field in preferred_fields:
                if field in available:
                    return field
    except Exception:
        pass
    # Fallback: peek first example for non-streaming datasets
    try:
        if hasattr(dataset, "__len__") and len(dataset) > 0:
            first = dataset[0]
            if isinstance(first, dict):
                available = set(first.keys())
                for field in preferred_fields:
                    if field in available:
                        return field
    except Exception:
        pass
    return preferred_fields[0]


def _load_dataset_with_split_fallback(builder_name, *, split, streaming=True, **kwargs):
    """
    Load dataset with a split fallback (train/validation/test may not exist).
    """
    try:
        return load_dataset(builder_name, split=split, streaming=streaming, **kwargs)
    except Exception:
        # Try a broader fallback order when the requested split is missing.
        fallback_orders = {
            "train": ("validation", "test"),
            "validation": ("train", "test"),
            "test": ("validation", "train"),
        }
        for alt in fallback_orders.get(split, ("train", "validation", "test")):
            try:
                return load_dataset(builder_name, split=alt, streaming=streaming, **kwargs)
            except Exception:
                continue
        # If still failing, re-raise the original error
        raise


def _ensure_streaming(dataset, builder_name, *, split, name=None):
    """Force streaming dataset to avoid full downloads."""
    if isinstance(dataset, IterableDataset):
        return dataset
    # Reload explicitly in streaming mode
    dataset = load_dataset(builder_name, name=name, split=split, streaming=True)
    if not isinstance(dataset, IterableDataset):
        raise RuntimeError(
            f"Expected streaming dataset for {builder_name} (name={name}, split={split})"
        )
    return dataset


def tokenize_texts(tokenizer, texts, max_length=None, min_length=None):
    """
    Tokenize texts and filter by length.

    Args:
        tokenizer: Tokenizer instance
        texts: List of text strings
        max_length: Maximum sequence length
        min_length: Minimum sequence length

    Returns:
        tokenized: List of tokenized sequences (as tensors)
        valid_indices: Indices of texts that passed length filtering
    """
    if max_length is None:
        max_length = Config.MAX_LEN
    if min_length is None:
        min_length = Config.MIN_LEN

    tokenized = []
    valid_indices = []

    for idx, text in enumerate(texts):
        # Tokenize
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_length)
        tokens = tokens.squeeze(0)  # Remove batch dimension

        # Filter by length
        if len(tokens) >= min_length:
            tokenized.append(tokens)
            valid_indices.append(idx)

    return tokenized, valid_indices


def sample_and_tokenize_dataset(
    dataset,
    text_field,
    tokenizer,
    n_samples,
    total_tokens=None,
    max_length=None,
    min_length=None,
    seed=None,
    span_mode="prefix"
):
    """
    Sample texts from dataset, tokenize, and filter by length.

    Args:
        dataset: HuggingFace dataset
        text_field: Name of text field
        tokenizer: Tokenizer instance
        n_samples: Number of samples to draw
        total_tokens: Token budget to collect (streaming)
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        seed: Random seed for sampling

    Returns:
        tokenized: List of tokenized sequences
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if total_tokens is not None and n_samples is not None:
        raise ValueError("Specify only one of n_samples or total_tokens.")
    if total_tokens is None and n_samples is None:
        raise ValueError("Either n_samples or total_tokens must be provided.")

    if total_tokens is not None:
        tokenized = []
        collected_tokens = 0
        texts_collected = 0
        min_len = min_length or Config.MIN_LEN
        max_len = max_length or Config.MAX_LEN

        iterable = dataset
        if hasattr(dataset, "shuffle"):
            try:
                iterable = dataset.shuffle(seed=seed) if seed is not None else dataset
            except Exception:
                iterable = dataset

        est_samples = max(1, total_tokens // max(1, min_len))
        max_iterations = est_samples * 10

        for iteration, example in enumerate(iterable):
            if iteration >= max_iterations:
                print(f"Warning: Reached max iterations ({max_iterations}) before collecting {total_tokens} tokens")
                break
            text = _extract_text(example, text_field)
            if text is None:
                continue
            texts_collected += 1
            if not isinstance(text, str) or len(text.strip()) == 0:
                continue

            # Pre-truncate in Python BEFORE tokenizing. HF fast tokenizers
            # tokenize the full string first, then apply truncation — a
            # pathological cost on long documents (e.g. Dolma book shards
            # with 600k+ char novels). At ~5-6 chars/token average, 20x
            # the target length gives a comfortable safety margin for the
            # fast path, while keeping the full string for `span_mode=middle`.
            if span_mode == "middle":
                tokens = tokenizer.encode(text, return_tensors='pt', truncation=False)
            else:
                text_cap = max_len * 20
                if len(text) > text_cap:
                    text = text[:text_cap]
                tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_len)
            tokens = tokens.squeeze(0)
            if span_mode == "middle" and len(tokens) > max_len:
                start = max(0, (len(tokens) - max_len) // 2)
                tokens = tokens[start:start + max_len]

            if len(tokens) >= min_len:
                tokenized.append(tokens)
                collected_tokens += len(tokens)
                if collected_tokens >= total_tokens:
                    break

            if texts_collected % 1000 == 0:
                print(f"  Collected {texts_collected} texts, {len(tokenized)} samples, {collected_tokens} tokens so far...")

        print(f"Final tokenized samples: {len(tokenized)} (tokens: {collected_tokens}, requested: {total_tokens})")
        return tokenized

    # Handle streaming/iterable datasets
    if isinstance(dataset, IterableDataset) or (hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__')):
        # Streaming dataset: iterate through examples
        # Keep sampling until we have enough valid tokenized samples
        tokenized = []
        texts_collected = 0
        max_iterations = n_samples * 10  # Safety limit to avoid infinite loops

        for iteration, example in enumerate(dataset):
            if iteration >= max_iterations:
                print(f"Warning: Reached max iterations ({max_iterations}) before collecting {n_samples} valid samples")
                break

            text = _extract_text(example, text_field)
            if text is not None:
                texts_collected += 1

                # Tokenize and filter immediately
                if not isinstance(text, str) or len(text.strip()) == 0:
                    continue
                # Pre-truncate in Python BEFORE tokenizing. HF fast tokenizers
                # tokenize the full string first, then apply truncation — a
                # pathological cost on long documents (e.g. Dolma book shards
                # with 600k+ char novels).
                max_len_tok = max_length or Config.MAX_LEN
                if span_mode == "middle":
                    tokens = tokenizer.encode(text, return_tensors='pt', truncation=False)
                else:
                    text_cap = max_len_tok * 20
                    if len(text) > text_cap:
                        text = text[:text_cap]
                    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_len_tok)
                tokens = tokens.squeeze(0)
                if span_mode == "middle" and len(tokens) > (max_length or Config.MAX_LEN):
                    max_len = max_length or Config.MAX_LEN
                    start = max(0, (len(tokens) - max_len) // 2)
                    tokens = tokens[start:start + max_len]

                # Check length filter
                if len(tokens) >= (min_length or Config.MIN_LEN):
                    tokenized.append(tokens)

                    if len(tokenized) >= n_samples:
                        break

                # Progress update for large datasets
                if texts_collected % 1000 == 0:
                    print(f"  Collected {texts_collected} texts, {len(tokenized)} valid samples so far...")
    else:
        # Regular dataset
        if len(dataset) > n_samples:
            # For regular datasets, we can sample more to account for filtering
            # Sample 20% more to account for potential filtering
            sample_size = int(n_samples * 1.2)
            sample_size = min(sample_size, len(dataset))
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            # Convert numpy int64 to Python int for dataset indexing
            texts = []
            for idx in indices:
                example = dataset[int(idx)]
                text = _extract_text(example, text_field)
                if text is not None:
                    texts.append(text)
        else:
            texts = []
            for idx in range(len(dataset)):
                example = dataset[idx]
                text = _extract_text(example, text_field)
                if text is not None:
                    texts.append(text)

        # Tokenize and filter
        if span_mode == "middle":
            tokenized = []
            for text in texts:
                tokens = tokenizer.encode(text, return_tensors='pt', truncation=False)
                tokens = tokens.squeeze(0)
                if len(tokens) > (max_length or Config.MAX_LEN):
                    max_len = max_length or Config.MAX_LEN
                    start = max(0, (len(tokens) - max_len) // 2)
                    tokens = tokens[start:start + max_len]
                if len(tokens) >= (min_length or Config.MIN_LEN):
                    tokenized.append(tokens)
        else:
            tokenized, _ = tokenize_texts(tokenizer, texts, max_length, min_length)

        # If we don't have enough after filtering, sample more
        if len(tokenized) < n_samples and len(dataset) > len(texts):
            remaining_needed = n_samples - len(tokenized)
            # Get remaining indices
            used_indices = set(indices if len(dataset) > n_samples else range(len(dataset)))
            remaining_indices = [i for i in range(len(dataset)) if i not in used_indices]

            if len(remaining_indices) > 0:
                additional_sample_size = min(remaining_needed * 2, len(remaining_indices))
                additional_indices = np.random.choice(remaining_indices, additional_sample_size, replace=False)
                additional_texts = []
                for idx in additional_indices:
                    example = dataset[int(idx)]
                    text = _extract_text(example, text_field)
                    if text is not None:
                        additional_texts.append(text)
                additional_tokenized, _ = tokenize_texts(tokenizer, additional_texts, max_length, min_length)
                tokenized.extend(additional_tokenized)

                # Trim to exactly n_samples if we have more
                if len(tokenized) > n_samples:
                    tokenized = tokenized[:n_samples]

    print(f"Final tokenized samples: {len(tokenized)} (requested: {n_samples})")
    return tokenized


def create_factual_prompts(dataset, text_field, n_prompts=5000, seed=None, min_prompt_chars=200, max_prompt_chars=500):
    """
    Generate factual prompts from dataset.

    Simple approach: extract short sentences or phrases that can serve as prompts.

    Args:
        dataset: HuggingFace dataset (can be streaming or regular)
        text_field: Name of text field
        n_prompts: Number of prompts to generate (None means use all available)
        seed: Random seed

    Returns:
        prompts: List of prompt strings
    """
    if seed is not None:
        np.random.seed(seed)

    prompts = []

    def _build_prompt(text):
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if len(text) == 0:
            return None
        # Prefer multi-sentence prefix to reach min_prompt_chars
        sentences = text.split('.')
        collected = []
        total_len = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            piece = s
            if not piece.endswith('.'):
                piece += '.'
            collected.append(piece)
            total_len += len(piece)
            if total_len >= min_prompt_chars:
                break
        if collected:
            prompt = " ".join(collected)
        else:
            prompt = text[:max_prompt_chars]
        # If still short, fall back to raw prefix (no sentence boundary)
        if len(prompt) < min_prompt_chars:
            prompt = text[:max_prompt_chars]
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]
        return prompt.strip()

    # Handle streaming/iterable datasets
    if isinstance(dataset, IterableDataset) or (hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__')):
        # Streaming dataset: iterate through examples
        if n_prompts is None:
            # Use all available samples
            for example in dataset:
                text = _extract_text(example, text_field)
                if text is not None:

                    prompt = _build_prompt(text)
                    if prompt and len(prompt) >= 10:
                        prompts.append(prompt)
        else:
            # Sample limited number
            n_samples = n_prompts * 2  # Sample more to account for filtering
            for example in dataset:
                text = _extract_text(example, text_field)
                if text is not None:

                    if not text or not isinstance(text, str):
                        continue

                    prompt = _build_prompt(text)

                    if prompt and len(prompt) >= 10:
                        prompts.append(prompt)
                        if len(prompts) >= n_prompts:
                            break

                    if len(prompts) >= n_samples:
                        break
    else:
        # Regular dataset
        if n_prompts is None:
            # Use all available samples
            indices = np.arange(len(dataset))
        else:
            # Sample limited number
            n_samples = min(len(dataset), n_prompts * 2)  # Sample more to account for filtering
            if len(dataset) > n_samples:
                indices = np.random.choice(len(dataset), n_samples, replace=False)
            else:
                indices = np.arange(len(dataset))

        for idx in indices:
            # Convert numpy int64 to Python int for dataset indexing
            text = _extract_text(dataset[int(idx)], text_field)

            if not text or not isinstance(text, str):
                continue

            prompt = _build_prompt(text)

            # Add prompt if valid
            if prompt and len(prompt) >= 10:
                prompts.append(prompt)
                if n_prompts is not None and len(prompts) >= n_prompts:
                    break

    print(f"Generated {len(prompts)} prompts from dataset")
    if len(prompts) == 0:
        print("Warning: No prompts generated! Check dataset and text_field.")
        # Debug: print first few examples
        if hasattr(dataset, '__len__') and len(dataset) > 0:
            print(f"Dataset size: {len(dataset)}")
            print(f"First example keys: {list(dataset[0].keys())}")
            sample_text = _extract_text(dataset[0], text_field)
            if sample_text is not None:
                print(f"Sample text (first 200 chars): {sample_text[:200] if isinstance(sample_text, str) else type(sample_text)}")

    return prompts


def batch_tokenize(tokenizer, texts, batch_size=32, max_length=None):
    """
    Tokenize texts in batches and pad to same length.

    Args:
        tokenizer: Tokenizer instance
        texts: List of text strings
        batch_size: Batch size for tokenization
        max_length: Maximum sequence length

    Returns:
        tokens: Padded token tensor, shape [batch_size, seq_len]
    """
    if max_length is None:
        max_length = Config.MAX_LEN

    token_batches = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        token_batches.append(tokens['input_ids'])

    return torch.cat(token_batches, dim=0)

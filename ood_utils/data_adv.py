"""
Data loading utilities for Adversarial OOD experiments.
"""

import hashlib
import logging
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import IterableDataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer

from ood_utils.data_utils import (
    _extract_text,
    _load_dataset_with_split_fallback,
    _pick_text_field,
    batch_tokenize,
    create_factual_prompts,
    load_id_dataset,
    sample_and_tokenize_dataset,
    tokenize_texts,
)

# Suppress datasets library warnings about deprecated scripts
logging.getLogger("datasets").setLevel(logging.ERROR)


def _is_script_blocked_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "dataset scripts are no longer supported" in msg or "requires arbitrary python code" in msg


def _infer_local_loader(files: List[str]) -> str:
    exts = {f.split(".")[-1].lower() for f in files if "." in f}
    if "parquet" in exts:
        return "parquet"
    if "jsonl" in exts:
        return "json"
    if "json" in exts:
        return "json"
    return "json"


def _load_local_dataset_from_env(env_key: str, split: str):
    """
    Load dataset from local files defined by env var.
    Env var can be a directory or a comma-separated list of files/globs.
    """
    import glob

    path_or_glob = os.environ.get(env_key)
    if not path_or_glob:
        return None
    parts = [p.strip() for p in path_or_glob.split(",") if p.strip()]
    files: List[str] = []
    for p in parts:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "*.*"), recursive=True))
        else:
            files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if any(f.endswith(ext) for ext in (".json", ".jsonl", ".parquet"))]
    files = sorted(set(files))
    if not files:
        return None
    loader = _infer_local_loader(files)
    ds = load_dataset(loader, data_files=files, split="train")
    return ds


def _download_hf_dataset_files(repo_id: str, patterns: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Download dataset repo snapshot via huggingface_hub and return matching data files.
    This avoids dataset scripts by loading raw files (json/jsonl/parquet) when available.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None
    try:
        cache_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return None

    import glob

    if patterns is None:
        patterns = [
            os.path.join(cache_dir, "**", "*.json"),
            os.path.join(cache_dir, "**", "*.jsonl"),
            os.path.join(cache_dir, "**", "*.parquet"),
        ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    files = sorted(set(files))
    return files if files else None


# --------------------------
# Adversarial datasets
# --------------------------

def _safe_get_str(example: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _ensure_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if isinstance(v, (str, int, float))]
    if isinstance(x, (tuple, set)):
        return [str(v) for v in x]
    return [str(x)]


def _find_first_text(obj, max_depth: int = 4) -> Optional[str]:
    if max_depth < 0:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        return s if s else None
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_first_text(v, max_depth - 1)
            if found:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_first_text(v, max_depth - 1)
            if found:
                return found
        return None
    return None


def _first_token_id(text: Optional[str], tokenizer: Optional[AutoTokenizer]) -> Optional[int]:
    if tokenizer is None or not isinstance(text, str) or not text.strip():
        return None
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        return None
    return int(ids[0])


def _normalize_binary_label(raw) -> Optional[bool]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw) != 0)
    if isinstance(raw, str):
        s = raw.strip().lower()
        pos = {
            "1",
            "true",
            "t",
            "yes",
            "y",
            "hallucinated",
            "hallucination",
            "is_hallucinated",
            "fake",
            "incorrect",
        }
        neg = {
            "0",
            "false",
            "f",
            "no",
            "n",
            "non-hallucinated",
            "non_hallucinated",
            "factual",
            "correct",
        }
        if s in pos:
            return True
        if s in neg:
            return False
    return None


def _halueval_record_kind(example: dict) -> Optional[str]:
    dialogue = _safe_get_str(example, ["dialogue", "history", "conversation", "chat_history"])
    query = _safe_get_str(example, ["query", "question", "instruction", "prompt", "input"])
    if dialogue:
        return "dialogue"
    if query:
        return "qa"
    return None


def _format_halueval_prompt_base(example: dict) -> Optional[str]:
    kind = _halueval_record_kind(example)
    if kind == "dialogue":
        dialogue = _safe_get_str(example, ["dialogue", "history", "conversation", "chat_history"])
        if dialogue:
            return f"{dialogue}\nAssistant:"
    if kind == "qa":
        query = _safe_get_str(example, ["query", "question", "instruction", "prompt", "input"])
        if query:
            return f"Question: {query}\nAnswer:"
    return None


def _extract_halueval_metadata(example: dict, tokenizer: Optional[AutoTokenizer]) -> Dict[str, object]:
    raw_label = example.get("label")
    if raw_label is None:
        raw_label = example.get("hallucination")
    if raw_label is None:
        raw_label = example.get("is_hallucinated")
    label = _normalize_binary_label(raw_label)
    response = _safe_get_str(example, ["response", "answer", "output", "assistant_response", "completion"])
    meta: Dict[str, object] = {
        "ood_type": "halu_eval",
        "hallucination_label": label,
        "response": response,
        "halueval_type": _halueval_record_kind(example),
    }
    provided = _first_token_id(response, tokenizer)
    if provided is not None:
        meta["provided_target_token_id"] = provided
    return meta


def _extract_jbb_behavior(example: dict) -> Optional[str]:
    behavior = _safe_get_str(example, ["behavior", "behavior_id", "behavior_name", "goal", "category"])
    if behavior:
        return behavior
    return None


def _format_jailbreak_prompt_base(example: dict) -> Optional[str]:
    prompt = _safe_get_str(example, ["prompt", "instruction", "query", "goal"])
    if not prompt:
        prompt = _find_first_text(example)
    if not prompt:
        return None
    return f"Instruction: {prompt}\nResponse:"


def _format_jailbreakhub_prompt_base(example: dict) -> Optional[str]:
    prompt = _safe_get_str(example, ["prompt"])
    if not prompt:
        prompt = _find_first_text(example)
    if not prompt:
        return None
    return f"Instruction: {prompt}\nResponse:"


def _extract_jailbreak_metadata(example: dict, tokenizer: Optional[AutoTokenizer]) -> Dict[str, object]:
    behavior = _extract_jbb_behavior(example)
    response = _safe_get_str(example, ["response", "answer", "output", "assistant_response", "completion"])
    meta: Dict[str, object] = {
        "ood_type": "jailbreakbench",
        "behavior": behavior,
        "response": response,
    }
    provided = _first_token_id(response, tokenizer)
    if provided is not None:
        meta["provided_target_token_id"] = provided
    return meta


def _stable_hash_float(text: str, seed: int = 2026) -> float:
    key = f"{seed}::{text}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(key).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _split_keep(split: str, key: str, train_ratio: float, seed: int = 2026) -> bool:
    if split not in {"train", "test", "validation", "val"}:
        return True
    frac = _stable_hash_float(key, seed=seed)
    if split == "train":
        return frac < train_ratio
    return frac >= train_ratio


def _halueval_split_key(example: dict) -> str:
    parts = [
        _safe_get_str(example, ["query", "question", "instruction", "prompt", "input"]),
        _safe_get_str(example, ["dialogue", "history", "conversation", "chat_history"]),
        _safe_get_str(example, ["response", "answer", "output", "assistant_response", "completion"]),
    ]
    joined = "\n".join([p for p in parts if isinstance(p, str) and p])
    if joined:
        return joined
    fallback = _find_first_text(example)
    return fallback if fallback else str(example)


def _jbb_split_key(example: dict) -> str:
    parts = [
        _extract_jbb_behavior(example),
        _safe_get_str(example, ["prompt", "instruction", "query", "goal"]),
        _safe_get_str(example, ["response", "answer", "output", "assistant_response", "completion"]),
    ]
    joined = "\n".join([p for p in parts if isinstance(p, str) and p])
    if joined:
        return joined
    fallback = _find_first_text(example)
    return fallback if fallback else str(example)


def _extract_jailbreakhub_metadata(example: dict, tokenizer: Optional[AutoTokenizer]) -> Dict[str, object]:
    response = _safe_get_str(example, ["response", "answer", "output", "assistant_response", "completion"])
    meta: Dict[str, object] = {
        "ood_type": "jailbreakhub",
        "response": response,
    }
    provided = _first_token_id(response, tokenizer)
    if provided is not None:
        meta["provided_target_token_id"] = provided
    return meta


def _jailbreakhub_split_key(example: dict) -> str:
    prompt = _safe_get_str(example, ["prompt"])
    if prompt:
        return prompt
    fallback = _find_first_text(example)
    return fallback if fallback else str(example)


def _make_split_aware_text_field(
    base_formatter: Callable[[dict], Optional[str]],
    split: str,
    split_key_fn: Callable[[dict], str],
    train_ratio: float,
    seed: int = 2026,
) -> Callable[[dict], Optional[str]]:
    def _text_field(example: dict) -> Optional[str]:
        if not isinstance(example, dict):
            return None
        key = split_key_fn(example)
        if not _split_keep(split, key=key, train_ratio=train_ratio, seed=seed):
            return None
        return base_formatter(example)

    return _text_field


class _RoundRobinIterable:
    """Round-robin iterable wrapper to combine heterogeneous iterable datasets."""

    def __init__(self, datasets: List):
        self._datasets = list(datasets)

    def __iter__(self):
        iterators = [iter(ds) for ds in self._datasets]
        while iterators:
            next_iterators = []
            for it in iterators:
                try:
                    ex = next(it)
                except StopIteration:
                    continue
                next_iterators.append(it)
                yield ex
            iterators = next_iterators


def _combine_datasets(parts: List):
    if not parts:
        raise RuntimeError("No dataset parts to combine.")
    if len(parts) == 1:
        return parts[0]
    return _RoundRobinIterable(parts)


def _load_halueval_dataset(configs: List[str]):
    parts = []
    for cfg in configs:
        ds = _load_dataset_with_split_fallback(
            "pminervini/HaluEval",
            name=cfg,
            split="data",
            streaming=True,
        )
        parts.append(ds)
    return _combine_datasets(parts)


def _load_jbb_dataset(source_split: str):
    if source_split == "both":
        ds_harm = _load_dataset_with_split_fallback(
            "JailbreakBench/JBB-Behaviors",
            name="behaviors",
            split="harmful",
            streaming=True,
        )
        ds_benign = _load_dataset_with_split_fallback(
            "JailbreakBench/JBB-Behaviors",
            name="behaviors",
            split="benign",
            streaming=True,
        )
        return interleave_datasets([ds_harm, ds_benign], probabilities=[0.5, 0.5])

    if source_split not in {"harmful", "benign"}:
        raise ValueError("JBB_SOURCE_SPLIT must be one of: harmful, benign, both")

    return _load_dataset_with_split_fallback(
        "JailbreakBench/JBB-Behaviors",
        name="behaviors",
        split=source_split,
        streaming=True,
    )


def _load_jailbreakhub_dataset():
    ds = _load_dataset_with_split_fallback(
        "walledai/JailbreakHub",
        split="train",
        streaming=False,
    )
    ds = ds.filter(lambda x: x.get("jailbreak") is True)
    return ds


def _parse_train_ratio(env_key: str, default_ratio: float) -> float:
    raw = os.environ.get(env_key)
    if raw is None:
        return float(default_ratio)
    try:
        ratio = float(raw)
    except Exception:
        return float(default_ratio)
    if ratio <= 0.0 or ratio >= 1.0:
        return float(default_ratio)
    return ratio


def load_ood_dataset(ood_type, split="train"):
    """
    Load adversarial OOD dataset.

    Supported:
      - halu_eval
      - jailbreakbench
      - jailbreakhub
    """
    split_seed = int(os.environ.get("ADV_PROMPT_SPLIT_SEED", "2026"))
    global_ratio = _parse_train_ratio("ADV_PROMPT_SPLIT_TRAIN_RATIO", 0.7)

    if ood_type == "truthfulqa":
        raise ValueError(
            "truthfulqa is excluded from Adversarial OOD in the current protocol. "
            "Use ood_set in {'halu_eval', 'jailbreakbench'}."
        )

    if ood_type == "halu_eval":
        ratio = _parse_train_ratio("HALUEVAL_PROMPT_SPLIT_TRAIN_RATIO", global_ratio)
        configs_raw = os.environ.get("HALUEVAL_CONFIGS", "qa,dialogue")
        configs = [c.strip() for c in configs_raw.split(",") if c.strip()]
        configs = [c for c in configs if c.lower() in {"qa", "dialogue"}]
        if not configs:
            configs = ["qa", "dialogue"]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset = _load_halueval_dataset(configs)
        except Exception as err:
            if _is_script_blocked_error(err):
                local_ds = _load_local_dataset_from_env("HALUEVAL_DATA", split)
                if local_ds is None:
                    raise RuntimeError(
                        "HaluEval dataset script is blocked. "
                        "Set HALUEVAL_DATA to a local .json/.jsonl/.parquet file or directory."
                    ) from err
                dataset = local_ds
            else:
                raise

        text_field = _make_split_aware_text_field(
            base_formatter=_format_halueval_prompt_base,
            split=split,
            split_key_fn=_halueval_split_key,
            train_ratio=ratio,
            seed=split_seed,
        )
        return dataset, text_field

    if ood_type == "jailbreakbench":
        ratio = _parse_train_ratio("JBB_PROMPT_SPLIT_TRAIN_RATIO", global_ratio)
        source_split = os.environ.get("JBB_SOURCE_SPLIT", "harmful").strip().lower()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset = _load_jbb_dataset(source_split)
        except Exception as err:
            if _is_script_blocked_error(err):
                local_ds = _load_local_dataset_from_env("JBB_DATA", split)
                if local_ds is None:
                    raise RuntimeError(
                        "JailbreakBench dataset script is blocked. "
                        "Set JBB_DATA to a local .json/.jsonl/.parquet file or directory."
                    ) from err
                dataset = local_ds
            else:
                raise

        text_field = _make_split_aware_text_field(
            base_formatter=_format_jailbreak_prompt_base,
            split=split,
            split_key_fn=_jbb_split_key,
            train_ratio=ratio,
            seed=split_seed,
        )
        return dataset, text_field

    if ood_type == "jailbreakhub":
        ratio = _parse_train_ratio("JAILBREAKHUB_PROMPT_SPLIT_TRAIN_RATIO", global_ratio)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset = _load_jailbreakhub_dataset()
        except Exception as err:
            if _is_script_blocked_error(err):
                local_ds = _load_local_dataset_from_env("JAILBREAKHUB_DATA", split)
                if local_ds is None:
                    raise RuntimeError(
                        "JailbreakHub dataset script is blocked. "
                        "Set JAILBREAKHUB_DATA to a local .json/.jsonl/.parquet file or directory."
                    ) from err
                dataset = local_ds
            else:
                raise

        text_field = _make_split_aware_text_field(
            base_formatter=_format_jailbreakhub_prompt_base,
            split=split,
            split_key_fn=_jailbreakhub_split_key,
            train_ratio=ratio,
            seed=split_seed,
        )
        return dataset, text_field

    raise ValueError(f"Unknown adversarial OOD dataset: {ood_type}")


def create_adversarial_prompts(
    dataset,
    ood_type: str,
    tokenizer: Optional[AutoTokenizer] = None,
    n_prompts: Optional[int] = 5000,
    seed: Optional[int] = None,
    include_choices: bool = True,
    text_field: Optional[Callable[[dict], Optional[str]]] = None,
):
    """
    Create adversarial prompts with metadata for target selection/reporting.

    Returns:
        prompts: List[str]
        metadata: List[dict]
    """
    del include_choices  # kept for backward compatibility

    rng = np.random.default_rng(seed)

    prompts: List[str] = []
    metadata: List[Dict[str, object]] = []

    def _maybe_add(example: dict):
        if ood_type == "truthfulqa":
            raise ValueError("truthfulqa is no longer supported in the adversarial protocol.")

        if ood_type == "halu_eval":
            prompt = text_field(example) if text_field is not None else _format_halueval_prompt_base(example)
            if not prompt:
                return
            prompts.append(prompt)
            metadata.append(_extract_halueval_metadata(example, tokenizer))
            return

        if ood_type == "jailbreakbench":
            prompt = text_field(example) if text_field is not None else _format_jailbreak_prompt_base(example)
            if not prompt:
                return
            prompts.append(prompt)
            metadata.append(_extract_jailbreak_metadata(example, tokenizer))
            return

        if ood_type == "jailbreakhub":
            prompt = text_field(example) if text_field is not None else _format_jailbreakhub_prompt_base(example)
            if not prompt:
                return
            prompts.append(prompt)
            metadata.append(_extract_jailbreakhub_metadata(example, tokenizer))
            return

        raise ValueError(f"Unknown adversarial OOD dataset: {ood_type}")

    # Handle streaming/iterable datasets
    if isinstance(dataset, IterableDataset) or (hasattr(dataset, "__iter__") and not hasattr(dataset, "__len__")):
        if n_prompts is None:
            for example in dataset:
                if isinstance(example, dict):
                    _maybe_add(example)
        else:
            for example in dataset:
                if isinstance(example, dict):
                    _maybe_add(example)
                    if len(prompts) >= n_prompts:
                        break
    else:
        if n_prompts is None:
            indices = np.arange(len(dataset))
        else:
            n_samples = min(len(dataset), int(n_prompts * 1.2))
            if len(dataset) > n_samples:
                indices = rng.choice(len(dataset), n_samples, replace=False)
            else:
                indices = np.arange(len(dataset))
        for idx in indices:
            ex = dataset[int(idx)]
            if isinstance(ex, dict):
                _maybe_add(ex)
                if n_prompts is not None and len(prompts) >= n_prompts:
                    break

    print(f"Generated {len(prompts)} adversarial prompts from dataset")
    return prompts, metadata


def sample_and_tokenize_dataset_by_token_budget(
    dataset,
    text_field: Callable[[dict], Optional[str]],
    tokenizer: AutoTokenizer,
    token_budget: int,
    max_length: int = 256,
    min_length: int = 1,
    seed: Optional[int] = None,
    max_samples: Optional[int] = None,
):
    """
    Build a tokenized pool until cumulative token count reaches token_budget.
    Returns:
        tokenized: List[torch.Tensor]
        total_tokens: int
    """
    if token_budget <= 0:
        return [], 0
    rng = np.random.default_rng(seed)

    tokenized: List[torch.Tensor] = []
    total_tokens = 0

    def _try_add(example: dict):
        nonlocal total_tokens
        if not isinstance(example, dict):
            return
        try:
            text = text_field(example)
        except Exception:
            return
        if not isinstance(text, str) or not text.strip():
            return
        ids = tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).squeeze(0)
        n_tok = int(ids.numel())
        if n_tok < int(min_length):
            return
        tokenized.append(ids)
        total_tokens += n_tok

    # Iterable / streaming datasets: single pass until budget
    if isinstance(dataset, IterableDataset) or (hasattr(dataset, "__iter__") and not hasattr(dataset, "__len__")):
        for ex in dataset:
            _try_add(ex)
            if total_tokens >= token_budget:
                break
            if max_samples is not None and len(tokenized) >= int(max_samples):
                break
        return tokenized, total_tokens

    # Map-style datasets: random traversal to avoid order bias
    n = len(dataset)
    if n == 0:
        return tokenized, total_tokens
    order = np.arange(n)
    rng.shuffle(order)
    for idx in order:
        _try_add(dataset[int(idx)])
        if total_tokens >= token_budget:
            break
        if max_samples is not None and len(tokenized) >= int(max_samples):
            break
    return tokenized, total_tokens

"""
Data loading utilities for Time-shift OOD experiments.
"""
import torch
import warnings
import logging
import os
import re
import glob
import hashlib
import gzip
from datasets import load_dataset, IterableDataset
try:
    from datasets import DatasetDict, IterableDatasetDict
    HAS_DATASET_DICTS = True
except Exception:
    HAS_DATASET_DICTS = False
from transformers import AutoTokenizer
import numpy as np
from config import Config
from ood_utils.data_utils import (
    load_id_dataset,
    _extract_text,
    _pick_text_field,
    _load_dataset_with_split_fallback,
    _ensure_streaming,
    tokenize_texts,
    sample_and_tokenize_dataset,
    create_factual_prompts,
    batch_tokenize,
)

# Suppress datasets library warnings about deprecated scripts
logging.getLogger("datasets").setLevel(logging.ERROR)


def _load_fineweb_with_safe_split(split: str):
    """
    Load FineWeb while avoiding train/test leakage via split fallback.

    For test, we allow fallback to validation only (never to train).
    """
    def _try_load(candidate_split: str):
        import time
        from huggingface_hub import HfFolder
        token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
        last_err = None
        for attempt in range(5):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ds = load_dataset(
                        "HuggingFaceFW/fineweb",
                        split=candidate_split,
                        streaming=True,
                        token=token,
                    )
                return _ensure_streaming(ds, "HuggingFaceFW/fineweb", split=candidate_split)
            except Exception as e:
                s = str(e)
                if "429" in s or "Too Many Requests" in s or "rate limit" in s.lower():
                    wait = 30 * (2 ** attempt)
                    print(f"  [FineWeb {candidate_split}] 429 rate-limit (attempt {attempt+1}/5). Waiting {wait}s...")
                    time.sleep(wait)
                    last_err = e
                    continue
                raise
        raise last_err

    def _has_heldout_split() -> bool:
        for heldout in ("test", "validation"):
            try:
                _try_load(heldout)
                return True
            except Exception:
                continue
        return False

    def _example_text_for_hash(example):
        if not isinstance(example, dict):
            return ""
        for key in ("text", "content", "document", "data"):
            v = example.get(key)
            if isinstance(v, str) and v:
                return v
        return str(example)

    def _apply_pseudo_split(ds, split_name: str):
        try:
            test_fraction = float(os.environ.get("FINEWEB_TEST_FRACTION", "0.1"))
        except Exception:
            test_fraction = 0.1
        test_fraction = min(max(test_fraction, 0.01), 0.5)
        threshold = int(test_fraction * 1000)

        def _in_test_bucket(ex):
            text = _example_text_for_hash(ex)
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % 1000
            return bucket < threshold

        if split_name in ("test", "validation"):
            return ds.filter(_in_test_bucket)
        if split_name == "train":
            return ds.filter(lambda ex: not _in_test_bucket(ex))
        return ds

    # 1) Prefer real requested split when available.
    try:
        return _try_load(split)
    except Exception:
        pass

    # 2) For held-out requests, try the other held-out split before pseudo split.
    if split == "test":
        try:
            return _try_load("validation")
        except Exception:
            pass
    elif split == "validation":
        try:
            return _try_load("test")
        except Exception:
            pass

    # 3) If no held-out split exists, construct deterministic pseudo split from train.
    try:
        train_ds = _try_load("train")
    except Exception as err:
        raise RuntimeError(
            f"Unable to load FineWeb split '{split}'. "
            f"Also failed to load fallback train split. Last error: {err}"
        )

    if not _has_heldout_split():
        return _apply_pseudo_split(train_ds, split)

    # 4) If held-out exists but requested split still failed, keep strict behavior.
    raise RuntimeError(
        f"Unable to load FineWeb split '{split}' while held-out splits exist. "
        "Please check dataset availability or pass a valid split."
    )


def _stable_hash_bucket(path: str, buckets: int = 1000) -> int:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def _split_dolma_files(files, split: str):
    """
    Deterministically split local Dolma files into disjoint train/test partitions.

    Optional env overrides:
      - DOLMA_TRAIN_GLOB
      - DOLMA_TEST_GLOB
      - DOLMA_TEST_FRACTION (default: 0.1)
    """
    train_glob_env = os.environ.get("DOLMA_TRAIN_GLOB")
    test_glob_env = os.environ.get("DOLMA_TEST_GLOB")

    if train_glob_env or test_glob_env:
        def _expand(env_value):
            pats = [p.strip() for p in (env_value or "").split(",") if p.strip()]
            out = []
            for pat in pats:
                out.extend(glob.glob(pat, recursive=True))
            return sorted(set(out))

        train_files = _expand(train_glob_env)
        test_files = _expand(test_glob_env)

        if split == "train":
            selected = train_files
        elif split == "test":
            selected = test_files
        elif split == "validation":
            selected = test_files
        else:
            selected = files

        if not selected:
            raise RuntimeError(
                f"No files selected for split='{split}' using "
                "DOLMA_TRAIN_GLOB / DOLMA_TEST_GLOB."
            )
        return selected

    # Default: deterministic hash-based split by file path.
    try:
        test_fraction = float(os.environ.get("DOLMA_TEST_FRACTION", "0.1"))
    except Exception:
        test_fraction = 0.1
    test_fraction = min(max(test_fraction, 0.01), 0.5)
    threshold = int(test_fraction * 1000)

    test_files = [f for f in files if _stable_hash_bucket(f) < threshold]
    if len(files) > 1 and len(test_files) == 0:
        test_files = [files[-1]]
    train_files = [f for f in files if f not in set(test_files)]
    if len(files) > 1 and len(train_files) == 0:
        train_files = files[:-1]
        test_files = [files[-1]]

    if split == "train":
        selected = train_files
    elif split == "test":
        selected = test_files
    elif split == "validation":
        selected = test_files
    else:
        selected = files

    if not selected:
        raise RuntimeError(
            f"No files available for split='{split}' after Dolma partitioning "
            f"(n_files={len(files)}, test_fraction={test_fraction})."
        )
    return selected


def _is_gzip_like(path: str) -> bool:
    name = os.path.basename(path)
    return name.endswith(".gz") or ".gz." in name


def _filter_valid_dolma_files(files):
    """
    Filter out corrupted gzip files to avoid runtime EOFError during streaming.

    Integrity checks are disabled by default for faster startup.
    Set DOLMA_VALIDATE_GZIP=1 to enable full gzip integrity checks.
    """
    validate = os.environ.get("DOLMA_VALIDATE_GZIP", "0").lower() not in ("0", "false", "no")
    if not validate:
        return files

    good = []
    bad = []
    for path in files:
        if not _is_gzip_like(path):
            good.append(path)
            continue
        try:
            # Full decompression check to catch truncated gzip streams.
            with gzip.open(path, "rb") as f:
                while f.read(1024 * 1024):
                    pass
            good.append(path)
        except Exception as err:
            bad.append((path, str(err)))

    if bad:
        print(f"Warning: filtered {len(bad)} corrupted gzip file(s) from Dolma split.")
        for path, msg in bad[:5]:
            print(f"  - {path}: {msg}")
        if len(bad) > 5:
            print(f"  ... and {len(bad) - 5} more")

    if not good:
        raise RuntimeError(
            "All selected Dolma files failed gzip integrity checks. "
            "Re-download files or disable checks with DOLMA_VALIDATE_GZIP=0."
        )
    return good




def load_ood_dataset(ood_type, split="train"):
    """
    Load OOD time-shift dataset.
    
    Args:
        ood_type: 'fineweb' or 'dolma_web'
        split: Dataset split to load ('train' for training/adaptation, 'test' for evaluation)
    
    Returns:
        dataset: HuggingFace dataset
        text_field: Name of the text field in the dataset
    """
    if ood_type == 'fineweb':
        # FineWeb (recent CC-derived web text)
        dataset = _load_fineweb_with_safe_split(split=split)
        text_field = _pick_text_field(dataset, ["text", "content", "document", "data"])
    elif ood_type == 'dolma_web':
        # Dolma Web subset (Common Crawl 2024-2025) via local files to avoid dataset scripts
        data_dir = os.environ.get("DATA_DIR", os.environ.get("DATA_ROOT", "./data") + "/dataset_cache/dolma")
        if not data_dir or not os.path.isdir(data_dir):
            raise RuntimeError(
                "dolma_web requires DATA_DIR with downloaded Dolma files. "
                "Download a subset of URLs into DATA_DIR and retry."
            )
        glob_env = os.environ.get("DOLMA_FILE_GLOB")
        if glob_env:
            patterns = [p.strip() for p in glob_env.split(",") if p.strip()]
        else:
            patterns = [
                os.path.join(data_dir, "*.json.gz"),
                os.path.join(data_dir, "*.json.gz.*"),
                os.path.join(data_dir, "*.jsonl.gz"),
                os.path.join(data_dir, "*.jsonl.zst"),
                os.path.join(data_dir, "*.jsonl"),
                os.path.join(data_dir, "**", "*.json.gz"),
                os.path.join(data_dir, "**", "*.json.gz.*"),
                os.path.join(data_dir, "**", "*.jsonl.gz"),
                os.path.join(data_dir, "**", "*.jsonl.zst"),
                os.path.join(data_dir, "**", "*.jsonl"),
            ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat, recursive=True))
        files = sorted(set(files))
        if not files:
            try:
                sample = sorted(os.listdir(data_dir))[:10]
            except Exception:
                sample = []
            raise RuntimeError(
                f"No Dolma .jsonl(.gz/.zst) files found under DATA_DIR={data_dir}. "
                "Download a subset of URLs into DATA_DIR, set DOLMA_FILE_GLOB, "
                f"or check files (sample: {sample})."
            )
        files_for_split = _split_dolma_files(files, split=split)
        files_for_split = _filter_valid_dolma_files(files_for_split)
        dataset = load_dataset(
            "json",
            data_files=files_for_split,
            split="train",
            streaming=True,
        )
        text_field = _pick_text_field(dataset, ["text", "content", "document", "data"])
    else:
        raise ValueError(
            f"Unknown OOD type: {ood_type}. Use 'fineweb' or 'dolma_web'"
        )
    
    return dataset, text_field

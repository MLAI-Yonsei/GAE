#!/usr/bin/env python3
"""
Domain-OOD prompt generator enforcing four difficulty rules.
"""
import os
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm

from text_utils import (
    normalize_text,
    split_segments,
    build_edgar_template,
    build_patent_template,
    is_bad_target,
    last80_contains,
    has_2hop,
    EDGAR_BAIT,
    PATENT_BAIT,
    EDGAR_LEADINS,
    PATENT_LEADINS,
)


def _extract_text(example: dict, field):
    if callable(field):
        return field(example)
    if isinstance(field, (list, tuple)):
        cur = example
        for k in field:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur
    if isinstance(field, str):
        return example.get(field)
    return None


def _load_patents(split: str = "train"):
    ds = load_dataset("NortheasternUniversity/big_patent", name="all", split=split, streaming=True)
    text_field = "description"
    return ds, text_field


def _load_edgar(split: str = "train"):
    ds = load_dataset("jlohding/sp500-edgar-10k", split=split, streaming=True)
    # item_1a only
    def _item_1a(ex):
        for key in ("item_1a", "item1a", "item_1A", "item1A", "risk_factors"):
            if key in ex and isinstance(ex[key], str):
                return ex[key]
        return ex.get("item_1") if isinstance(ex, dict) else None
    return ds, _item_1a


def _tokenize(text: str, tokenizer):
    return tokenizer.encode(text, add_special_tokens=False)


def _build_example(
    dataset: str,
    segment: str,
    tokenizer,
    max_len: int,
    rng: random.Random,
    stats: Dict[str, int],
):
    # Template
    if dataset == "edgar":
        template, meta = build_edgar_template(segment)
        leadins = EDGAR_LEADINS
        bait_phrases = [meta["bait"]] + EDGAR_BAIT
        leadin_phrase = meta["lead_in"]
    else:
        template, meta = build_patent_template(segment)
        leadins = PATENT_LEADINS
        bait_phrases = [meta["bait"]] + PATENT_BAIT
        leadin_phrase = meta["lead_in"]

    # Ensure 2-hop within template
    if not has_2hop(template):
        stats["rule2_fail"] += 1
        return None

    # Pick continuation word to make content target
    continuation_candidates = ["significant", "material", "elevated", "reduced", "enhanced"]
    continuation = rng.choice(continuation_candidates)

    # Build tokens
    segment_tokens = _tokenize(segment, tokenizer)
    template_tokens = _tokenize(template, tokenizer)
    cont_tokens = _tokenize(" " + continuation, tokenizer)
    if not cont_tokens:
        stats["target_bad"] += 1
        return None

    required_prefix = max(0, (max_len - 1) - (len(template_tokens) - 1))
    if len(segment_tokens) == 0:
        return None
    # repeat segment tokens if needed to reach required length
    prefix_tokens = []
    while len(prefix_tokens) < required_prefix:
        prefix_tokens.extend(segment_tokens)
        if len(prefix_tokens) > required_prefix:
            prefix_tokens = prefix_tokens[-required_prefix:]

    leadin_end_idx = len(prefix_tokens) + len(template_tokens) - 1
    full_tokens = prefix_tokens + template_tokens + cont_tokens
    if leadin_end_idx + 1 >= len(full_tokens):
        stats["target_bad"] += 1
        return None

    start = leadin_end_idx - (max_len - 1)
    if start < 0:
        stats["window_fail"] += 1
        return None

    context_ids = full_tokens[start:start + max_len]
    if len(context_ids) != max_len:
        stats["window_fail"] += 1
        return None

    target_id = full_tokens[start + max_len]
    target_str = tokenizer.decode([target_id])
    if is_bad_target(target_str):
        stats["target_bad"] += 1
        return None

    last80 = tokenizer.decode(context_ids[-80:])
    # Rule checks
    has_bait = any(p in last80.lower() for p in bait_phrases)
    has_override = any(p in last80.lower() for p in ["however", "notwithstanding", "except when", "provided that", "unless"])
    has_leadin = any(p in last80.lower() for p in leadins)
    has_twohop = has_2hop(last80)
    last80_ok = has_bait and has_override and has_leadin and has_twohop
    if not last80_ok:
        stats["rule4_fail"] += 1
        return None

    final_text = tokenizer.decode(full_tokens)
    return {
        "dataset": dataset,
        "source_id": None,
        "final_text": final_text,
        "context_ids": context_ids,
        "position": max_len - 1,
        "target_next_token_id": int(target_id),
        "target_next_token_str": target_str,
        "flags": {
            "has_bait": has_bait,
            "has_override": has_override,
            "has_2hop": has_twohop,
            "has_leadin": has_leadin,
            "last80_ok": last80_ok,
        },
        "meta": {
            "lead_in": leadin_phrase,
            "override": meta["override"],
        },
    }


def _iter_dataset(ds):
    if isinstance(ds, IterableDataset):
        return ds
    if hasattr(ds, "to_iterable_dataset"):
        return ds.to_iterable_dataset()
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--n_total", type=int, default=50000)
    parser.add_argument("--model", type=str, default="gpt2",
                        choices=["gpt2", "pythia-410m", "pythia-1.4b"],
                        help="Model family to select tokenizer")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Override tokenizer name (HF hub id)")
    parser.add_argument("--ood_set", type=str, default="all",
                        choices=["all", "edgar", "patents"],
                        help="Which domain OOD set to generate")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Reduce background thread activity to avoid exit-time crashes.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.tokenizer is not None:
        tok_name = args.tokenizer
    else:
        tok_name = {
            "gpt2": "gpt2",
            "pythia-410m": "EleutherAI/pythia-410m",
            "pythia-1.4b": "EleutherAI/pythia-1.4b",
        }.get(args.model, "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.model_max_length = int(1e9)
    except Exception:
        pass

    if args.ood_set == "all":
        per_ds = args.n_total // 2
        targets = {"edgar": per_ds, "patents": args.n_total - per_ds}
        datasets = {
            "edgar": _load_edgar("train"),
            "patents": _load_patents("train"),
        }
    else:
        targets = {args.ood_set: args.n_total}
        datasets = {
            args.ood_set: _load_edgar("train") if args.ood_set == "edgar" else _load_patents("train"),
        }

    per_dataset_examples = {k: [] for k in datasets.keys()}
    diagnostics = {"rule2_fail": 0, "rule4_fail": 0, "target_bad": 0, "window_fail": 0}

    for ds_name, (ds, text_field) in datasets.items():
        ds_iter = _iter_dataset(ds)
        count = 0
        pbar = tqdm(total=targets[ds_name], desc=f"{ds_name}", unit="ex")
        try:
            for ex in ds_iter:
                if count >= targets[ds_name]:
                    break
                text = _extract_text(ex, text_field)
                if not isinstance(text, str) or not text.strip():
                    continue
                text = normalize_text(text)
                segments = split_segments(text, min_chars=300, max_chars=2000)
                if not segments:
                    continue
                rng.shuffle(segments)
                for seg in segments[:3]:
                    example = _build_example(ds_name, seg, tokenizer, args.max_len, rng, diagnostics)
                    if example is None:
                        continue
                    example["source_id"] = ex.get("id") if isinstance(ex, dict) else None
                    per_dataset_examples[ds_name].append(example)
                    count += 1
                    pbar.update(1)
                    break
        finally:
            pbar.close()

    # diagnostics
    total = sum(len(v) for v in per_dataset_examples.values())
    print("Acceptance:", total, "/", args.n_total)
    print("Per-dataset:", {k: len(v) for k, v in per_dataset_examples.items()})
    print("Rule fails:", diagnostics)
    if total > 0:
        flat = [ex for v in per_dataset_examples.values() for ex in v]
        for ex in rng.sample(flat, k=min(5, len(flat))):
            tail = tokenizer.decode(ex["context_ids"][-120:])
            print("---")
            print("dataset:", ex["dataset"])
            print("tail120:", tail)
            print("target:", ex["target_next_token_str"])

    if args.dry_run:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds_name, rows in per_dataset_examples.items():
        out_path = out_dir / f"{ds_name}_{args.model}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
    # Force exit to avoid PyGILState_Release crash in some environments.
    os._exit(0)

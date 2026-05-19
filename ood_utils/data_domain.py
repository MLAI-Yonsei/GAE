"""
Data loading utilities for Domain-OOD experiments.
"""
import torch
import warnings
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from config import Config
from ood_utils.data_utils import (
    load_id_dataset,
    _extract_text,
    _pick_text_field,
    _load_dataset_with_split_fallback,
    tokenize_texts,
    sample_and_tokenize_dataset,
    create_factual_prompts,
    batch_tokenize,
)

# Suppress datasets library warnings about deprecated scripts
logging.getLogger("datasets").setLevel(logging.ERROR)




def _is_script_blocked_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "dataset scripts are no longer supported" in msg or "requires arbitrary python code" in msg


def load_ood_dataset(ood_type, split="train"):
    """
    Load OOD domain dataset.
    
    Args:
        ood_type: 'patents', 'edgar', 'govreport', 'subtitles', or 'standards'
        split: Dataset split to load ('train' for training/adaptation, 'test' for evaluation)
    
    Returns:
        dataset: HuggingFace dataset
        text_field: Name of the text field in the dataset
    """
    if ood_type == 'patents':
        # Patents (technical-legal domain)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = _load_dataset_with_split_fallback(
                "NortheasternUniversity/big_patent",
                name="all",
                split=split,
                streaming=True,
            )
        # Use description only (exclude claims/abstract)
        text_field = _pick_text_field(dataset, ["description", "text"])

    elif ood_type == 'edgar':
        # SEC 10-K filings (financial/regulatory domain)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = _load_dataset_with_split_fallback(
                "jlohding/sp500-edgar-10k",
                split=split,
                streaming=True,
            )
        text_field = _pick_text_field(dataset, ["text", "item_1", "item1", "content", "filing_text"])

    elif ood_type == 'govreport':
        # US Government Reports (GAO/CRS policy analysis)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = _load_dataset_with_split_fallback(
                "ccdv/govreport-summarization",
                split=split,
                streaming=True,
            )
        text_field = "report"

    elif ood_type == 'standards':
        # Standards & specifications (normative technical domain)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = _load_dataset_with_split_fallback(
                "sidereior/wcg-standards-adherence",
                split=split,
                streaming=True,
            )

        def _standards_text(example):
            if not isinstance(example, dict):
                return None
            data = example.get("data")
            if isinstance(data, str):
                return data
            if isinstance(data, dict):
                for key in ("text", "content", "standard", "document", "guideline"):
                    if key in data and isinstance(data[key], str):
                        return data[key]
            # Fallback to top-level fields if present
            for key in ("text", "content", "standard", "document", "guideline"):
                if key in example and isinstance(example[key], str):
                    return example[key]
            return None

        text_field = _standards_text

    else:
        raise ValueError(
            f"Unknown OOD type: {ood_type}. Use 'patents', 'edgar', 'govreport', or 'standards'"
        )
    
    return dataset, text_field

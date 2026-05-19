import re
import random
import string
from typing import List, Tuple, Dict

STOP_TOKENS = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were","be","been","it","this","that"
}

OVERRIDE_CONNECTORS = ["however", "notwithstanding", "except when", "provided that", "unless"]

EDGAR_LEADINS = ["expected to be", "is likely to be", "will be", "may become"]
PATENT_LEADINS = ["is configured to", "thereby", "such that the output", "the controller then"]

EDGAR_BAIT = ["may be adversely affected", "could materially affect", "risk", "uncertainty", "adversely"]
PATENT_BAIT = ["at least one", "triggers an alert", "in some embodiments", "configured to", "threshold"]

EDGAR_KEYWORDS = ["risk", "uncertainty", "adverse", "material", "affect", "liquidity", "regulatory", "market", "interest", "currency", "credit"]
PATENT_KEYWORDS = ["sensor", "signal", "threshold", "detect", "anomaly", "alert", "controller", "module"]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_segments(text: str, min_chars: int = 300, max_chars: int = 2000) -> List[str]:
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    segments = []
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        if len(b) < min_chars:
            continue
        if len(b) > max_chars:
            # sentence pack
            sentences = re.split(r"(?<=[.!?])\s+", b)
            cur = []
            cur_len = 0
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if cur_len + len(s) + 1 > max_chars and cur:
                    segments.append(" ".join(cur).strip())
                    cur = [s]
                    cur_len = len(s)
                else:
                    cur.append(s)
                    cur_len += len(s) + 1
            if cur:
                segments.append(" ".join(cur).strip())
        else:
            segments.append(b)
    return segments


def _extract_clause(segment: str, keywords: List[str], fallback: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", segment)
    random.shuffle(sents)
    for s in sents:
        ls = s.lower()
        if any(k in ls for k in keywords):
            return s.strip()
    return fallback


def enforce_2hop(text: str, dataset: str) -> str:
    if "if" in text and ("only if" in text or "unless" in text):
        return text
    if dataset == "edgar":
        return text + " if demand decreases only if supply constraints persist"
    return text + " if the signal exceeds the threshold unless the calibration module is active"


def build_edgar_template(segment: str) -> Tuple[str, Dict[str, str]]:
    X = _extract_clause(
        segment,
        EDGAR_KEYWORDS,
        "market conditions deteriorate and liquidity tightens"
    )
    X = enforce_2hop(X, "edgar")
    Y = _extract_clause(
        segment,
        ["hedge", "diversif", "mitigat", "insurance", "offset", "reduce"],
        "we have mitigation measures and diversified exposures"
    )
    override = "however"
    bait = "may be adversely affected"
    lead_in = "expected to be"
    template = f"Our business may be adversely affected by {X}; {override}, {Y}, and therefore the impact is {lead_in}"
    return template, {"override": override, "bait": bait, "lead_in": lead_in}


def build_patent_template(segment: str) -> Tuple[str, Dict[str, str]]:
    X = _extract_clause(
        segment,
        PATENT_KEYWORDS,
        "sensor detects an anomaly"
    )
    X = enforce_2hop(X, "patent")
    Y = _extract_clause(
        segment,
        ["calibrat", "diagnostic", "transient", "filter"],
        "the calibration module is active"
    )
    override = "except when"
    bait = "triggers an alert"
    lead_in = "is configured to"
    template = f"The system triggers an alert when at least one {X}, {override} {Y}, in which case the controller {lead_in}"
    return template, {"override": override, "bait": bait, "lead_in": lead_in}


def pick_leadin_variant(dataset: str) -> str:
    return random.choice(EDGAR_LEADINS if dataset == "edgar" else PATENT_LEADINS)


def pick_override_variant() -> str:
    return random.choice(OVERRIDE_CONNECTORS)


def is_bad_target(token_str: str) -> bool:
    s = token_str.strip().lower()
    if s == "":
        return True
    if all(ch in string.punctuation for ch in s):
        return True
    if s in STOP_TOKENS:
        return True
    return False


def last80_contains(text: str, phrases: List[str]) -> bool:
    lt = text.lower()
    return all(p in lt for p in phrases)


def has_2hop(text: str) -> bool:
    t = text.lower()
    return ("if" in t and "only if" in t) or ("if" in t and "unless" in t)


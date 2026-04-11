"""Input/output replication similarity scoring for worm propagation analysis.

This module is dependency-free and provides BLEU + ROUGE-L style similarity
signals to detect when model output mirrors worm-like input payloads.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]{0,64}\|>|\[/?INST\]|{{|}}|[A-Za-z0-9_]+")


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return str(text)


def _tokenize(text: str | None, max_tokens: int = 2000) -> List[str]:
    normalized = _normalize_text(text).lower()
    tokens = _SPECIAL_TOKEN_RE.findall(normalized)
    if max_tokens > 0 and len(tokens) > max_tokens:
        return tokens[:max_tokens]
    return tokens


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(reference: Sequence[str], candidate: Sequence[str], n: int) -> float:
    cand_ngrams = _ngrams(candidate, n)
    if not cand_ngrams:
        return 0.0

    ref_counts = Counter(_ngrams(reference, n))
    cand_counts = Counter(cand_ngrams)
    overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
    total = len(cand_ngrams)

    # Smooth only higher-order n-grams; keep unigram precision strict.
    if overlap == 0 and n >= 2:
        return 1.0 / (total + 1.0)
    return overlap / float(total)


def bleu_score(reference: Sequence[str], candidate: Sequence[str], max_order: int = 4) -> float:
    if not reference or not candidate:
        return 0.0

    max_order = max(1, int(max_order))
    precisions = [_modified_precision(reference, candidate, n) for n in range(1, max_order + 1)]

    # If unigram overlap is zero, texts are unrelated.
    if precisions[0] <= 0.0:
        return 0.0

    log_mean = sum(math.log(max(p, 1e-12)) for p in precisions) / float(max_order)
    geo_mean = math.exp(log_mean)

    ref_len = len(reference)
    cand_len = len(candidate)
    if cand_len == 0:
        return 0.0

    brevity_penalty = 1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / float(cand_len)))
    return max(0.0, min(1.0, brevity_penalty * geo_mean))


def _lcs_length(a: Sequence[str], b: Sequence[str], max_cells: int = 4_000_000) -> int:
    if not a or not b:
        return 0

    # Guard against quadratic blowups on very long inputs.
    if len(a) * len(b) > max_cells:
        cap = max(1, int(math.sqrt(max_cells)))
        a = a[:cap]
        b = b[:cap]

    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, prev
    return prev[-1]


def rouge_l_score(reference: Sequence[str], candidate: Sequence[str]) -> float:
    if not reference or not candidate:
        return 0.0
    lcs = _lcs_length(reference, candidate)
    if lcs == 0:
        return 0.0

    precision = lcs / float(len(candidate))
    recall = lcs / float(len(reference))
    if precision + recall <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (2.0 * precision * recall) / (precision + recall)))


def replication_similarity(
    input_text: str | None,
    output_text: str | None,
    *,
    bleu_weight: float = 0.4,
    rouge_weight: float = 0.6,
    max_tokens: int = 2000,
) -> Dict[str, float]:
    """Return BLEU, ROUGE-L, and weighted combined replication similarity."""
    in_tokens = _tokenize(input_text, max_tokens=max_tokens)
    out_tokens = _tokenize(output_text, max_tokens=max_tokens)

    if not in_tokens or not out_tokens:
        return {
            "bleu": 0.0,
            "rouge_l": 0.0,
            "combined": 0.0,
            "input_tokens": float(len(in_tokens)),
            "output_tokens": float(len(out_tokens)),
        }

    bleu = bleu_score(in_tokens, out_tokens)
    rouge_l = rouge_l_score(in_tokens, out_tokens)

    bleu_weight = max(0.0, float(bleu_weight))
    rouge_weight = max(0.0, float(rouge_weight))
    denom = max(1e-9, bleu_weight + rouge_weight)
    combined = ((bleu_weight * bleu) + (rouge_weight * rouge_l)) / denom

    return {
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge_l, 4),
        "combined": round(max(0.0, min(1.0, combined)), 4),
        "input_tokens": float(len(in_tokens)),
        "output_tokens": float(len(out_tokens)),
    }


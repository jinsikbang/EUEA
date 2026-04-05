import os
import json
import re
from collections import defaultdict
from typing import Any

try:
    from pycocoevalcap.tokenize.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    _COCO_AVAILABLE = True
except ImportError:
    _COCO_AVAILABLE = False


def _normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, and collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_exact_match(predictions: list[str], references: list[list[str]]) -> float:
    """Compute Exact Match (EM) accuracy.

    Args:
        predictions: List of predicted answer strings.
        references: List of lists of reference answer strings.

    Returns:
        Exact match accuracy in the range [0, 1].
    """
    correct = 0
    for pred, refs in zip(predictions, references):
        pred_norm = _normalize_answer(pred)
        if any(_normalize_answer(r) == pred_norm for r in refs):
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def compute_cider(
    predictions: list[str],
    references: list[list[str]],
) -> float:
    """Compute CIDEr score using pycocoevalcap.

    Args:
        predictions: List of predicted strings, one per sample.
        references: List of lists of reference strings, one list per sample.

    Returns:
        CIDEr score.
    """
    if not _COCO_AVAILABLE:
        raise ImportError(
            "pycocoevalcap is required for CIDEr computation. "
            "Install it with: pip install pycocoevalcap"
        )
    gts: dict[int, list[dict]] = {}
    res: dict[int, list[dict]] = {}
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        gts[i] = [{"caption": r} for r in refs]
        res[i] = [{"caption": pred}]

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score


def compute_bleu(
    predictions: list[str],
    references: list[list[str]],
    n: int = 4,
) -> dict[str, float]:
    """Compute BLEU-1 through BLEU-n scores.

    Args:
        predictions: List of predicted strings.
        references: List of lists of reference strings.
        n: Maximum n-gram order (1–4).

    Returns:
        Dictionary mapping 'bleu_1' … 'bleu_<n>' to their scores.
    """
    if not _COCO_AVAILABLE:
        raise ImportError(
            "pycocoevalcap is required for BLEU computation. "
            "Install it with: pip install pycocoevalcap"
        )
    gts: dict[int, list[dict]] = {}
    res: dict[int, list[dict]] = {}
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        gts[i] = [{"caption": r} for r in refs]
        res[i] = [{"caption": pred}]

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorer = Bleu(n)
    score, _ = scorer.compute_score(gts, res)
    return {f"bleu_{i + 1}": s for i, s in enumerate(score)}


def compute_meteor(
    predictions: list[str],
    references: list[list[str]],
) -> float:
    """Compute METEOR score.

    Args:
        predictions: List of predicted strings.
        references: List of lists of reference strings.

    Returns:
        METEOR score.
    """
    if not _COCO_AVAILABLE:
        raise ImportError(
            "pycocoevalcap is required for METEOR computation. "
            "Install it with: pip install pycocoevalcap"
        )
    gts: dict[int, list[dict]] = {}
    res: dict[int, list[dict]] = {}
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        gts[i] = [{"caption": r} for r in refs]
        res[i] = [{"caption": pred}]

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorer = Meteor()
    score, _ = scorer.compute_score(gts, res)
    return score


def compute_metrics(
    predictions: list[str],
    references: list[list[str]],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute all requested evaluation metrics in one call.

    Args:
        predictions: List of predicted answer strings.
        references: List of lists of reference answer strings.
        metrics: Subset of ``["em", "cider", "bleu", "meteor"]`` to compute.
                 When *None* all metrics are computed.

    Returns:
        Dictionary mapping metric names to their scores.
    """
    if metrics is None:
        metrics = ["em", "cider", "bleu", "meteor"]

    results: dict[str, float] = {}

    if "em" in metrics:
        results["exact_match"] = compute_exact_match(predictions, references)

    if "cider" in metrics:
        results["cider"] = compute_cider(predictions, references)

    if "bleu" in metrics:
        results.update(compute_bleu(predictions, references))

    if "meteor" in metrics:
        results["meteor"] = compute_meteor(predictions, references)

    return results

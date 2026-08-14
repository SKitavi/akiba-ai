"""Evaluation metric reporting for binary credit-risk classification.

Produces a unified metrics dict covering accuracy, AUC-ROC, precision, recall,
F1, and a majority-class baseline comparison — all in one call.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def generate_classification_report(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return precision, recall, F1, AUC-ROC, and baseline comparison metrics.

    Args:
        y_true:    Ground-truth binary labels (0 = non-default, 1 = default).
        y_score:   Model-predicted probability of default in ``[0, 1]``.
        threshold: Decision threshold for converting probabilities to class
                   labels. Defaults to ``0.5``.

    Returns:
        Dictionary with the following keys:

        * ``n_samples``          – total sample count
        * ``positive_rate``      – fraction of positive labels in ``y_true``
        * ``threshold``          – threshold used for label binarisation
        * ``accuracy``           – classification accuracy at ``threshold``
        * ``baseline_accuracy``  – majority-class baseline accuracy (always
                                   predicting the most frequent class)
        * ``beats_baseline``     – bool: model accuracy > baseline accuracy
        * ``auc_roc``            – area under the ROC curve
        * ``precision``          – positive-class precision at ``threshold``
        * ``recall``             – positive-class recall at ``threshold``
        * ``f1``                 – positive-class F1 at ``threshold``
        * ``tp`` / ``fp`` / ``tn`` / ``fn`` – confusion-matrix counts

    Raises:
        ValueError: If ``y_true`` and ``y_score`` have different lengths or
                    if ``y_true`` contains values other than 0 and 1.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    if len(y_true_arr) != len(y_score_arr):
        raise ValueError(
            f"y_true (len={len(y_true_arr)}) and y_score (len={len(y_score_arr)}) "
            "must have the same length."
        )
    if not set(y_true_arr.tolist()).issubset({0, 1}):
        raise ValueError("y_true must contain only binary values 0 and 1.")

    n = len(y_true_arr)
    positive_rate = float(y_true_arr.mean())

    # --- Binarise scores at threshold ---
    y_pred = (y_score_arr >= threshold).astype(int)

    # --- Confusion matrix counts ---
    tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_arr == 1)).sum())

    # --- Per-metric computation ---
    accuracy = float((tp + tn) / n)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # --- Majority-class baseline ---
    majority_class = int(positive_rate >= 0.5)
    baseline_accuracy = float(
        (y_true_arr == majority_class).mean()
    )

    # --- AUC-ROC via trapezoidal rule (no sklearn dependency) ---
    auc_roc = _compute_auc_roc(y_true_arr, y_score_arr)

    return {
        "n_samples": n,
        "positive_rate": round(positive_rate, 4),
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "beats_baseline": bool(accuracy > baseline_accuracy),
        "auc_roc": round(auc_roc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC via the trapezoidal rule over all unique thresholds."""
    # Sort by descending score
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration
    return float(np.trapz(tpr, fpr))

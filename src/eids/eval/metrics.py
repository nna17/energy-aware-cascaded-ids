"""Detection metrics. Macro-F1 and minority recall are the headline numbers; accuracy is secondary."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support


def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                 minority: list[int], benign_index: int | None) -> tuple[float, float, float]:
    """(macro-F1, minority recall, benign FPR) from a confusion matrix; ~50x faster than sklearn for search loops.

    Matches `compute_metrics` (macro-F1 over classes present in y_true, zero_division=0).
    """
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    macro_f1 = float(f1[support > 0].mean())
    min_rec = float(recall[minority].mean()) if minority else float("nan")
    fpr = float(1 - cm[benign_index, benign_index] / support[benign_index]) if benign_index is not None else float("nan")
    return macro_f1, min_rec, fpr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                    minority: list[int], benign_index: int | None) -> dict:
    labels = np.arange(n_classes)
    _, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    present = support > 0
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels[present], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "minority_recall": float(np.mean(recall[minority])) if minority else float("nan"),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
    }
    if benign_index is not None:
        # Binary view: benign flows raised as any attack, and attacks raised as any attack.
        benign = y_true == benign_index
        out["false_positive_rate"] = float(np.mean(y_pred[benign] != benign_index)) if benign.any() else float("nan")
        out["detection_rate"] = float(np.mean(y_pred[~benign] != benign_index)) if (~benign).any() else float("nan")
    return out

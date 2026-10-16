import numpy as np
import pytest

from eids.distill.class_weights import class_weights
from eids.eval.metrics import compute_metrics


def test_minority_recall_exposes_rare_class_collapse():
    y_true = np.array([0] * 98 + [1, 1])
    y_pred = np.zeros(100, dtype=int)  # predicts benign everywhere
    m = compute_metrics(y_true, y_pred, n_classes=2, minority=[1], benign_index=0)
    assert m["accuracy"] == pytest.approx(0.98)
    assert m["minority_recall"] == 0.0
    assert m["false_positive_rate"] == 0.0
    assert m["detection_rate"] == 0.0


def test_binary_rates_ignore_confusion_between_attacks():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([1, 0, 2, 2])  # one benign false alarm; attack 1 called attack 2 still counts as detected
    m = compute_metrics(y_true, y_pred, n_classes=3, minority=[], benign_index=0)
    assert m["false_positive_rate"] == 0.5
    assert m["detection_rate"] == 1.0


def test_class_weights_favour_rare_classes_and_average_to_one():
    counts = np.array([1000, 10])
    for method in ("inverse_frequency", "effective_number"):
        w = class_weights(counts, method).numpy()
        assert w[1] > w[0]
        assert (w * counts).sum() / counts.sum() == pytest.approx(1.0)

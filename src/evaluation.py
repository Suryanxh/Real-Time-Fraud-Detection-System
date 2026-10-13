"""Validation-based decision-threshold selection."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    minimum_precision: float,
) -> tuple[float, bool]:
    """Choose the threshold with highest recall that meets a precision target."""
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Cannot select a threshold without score variation.")

    eligible = np.flatnonzero(precisions[:-1] >= minimum_precision)
    if eligible.size == 0:
        best_precision_index = int(np.argmax(precisions[:-1]))
        return float(thresholds[best_precision_index]), False

    best_index = int(eligible[np.argmax(recalls[eligible])])
    return float(thresholds[best_index]), True

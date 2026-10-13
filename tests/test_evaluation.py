import numpy as np

from src.evaluation import select_threshold


def test_select_threshold_meets_precision_target_and_maximizes_recall():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.85, 0.95])

    threshold, target_met = select_threshold(labels, scores, minimum_precision=0.9)

    assert target_met
    assert threshold == 0.85


def test_select_threshold_reports_when_target_is_unattainable():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    _, target_met = select_threshold(labels, scores, minimum_precision=0.9)

    assert not target_met

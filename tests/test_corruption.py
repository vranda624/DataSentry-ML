import pandas as pd
from datasentry.corruption import inject_corruptions, cell_detection_metrics


def test_corruption_is_reproducible():
    df = pd.DataFrame({"x": list(range(100)), "category": ["a"] * 100})
    corrupt1, labels1 = inject_corruptions(df, seed=7, fraction=0.05)
    corrupt2, labels2 = inject_corruptions(df, seed=7, fraction=0.05)
    pd.testing.assert_frame_equal(corrupt1, corrupt2)
    pd.testing.assert_frame_equal(labels1, labels2)


def test_perfect_predictions_score_one():
    truth = pd.DataFrame([[0, 1], [1, 0]])
    metrics = cell_detection_metrics(truth, truth)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

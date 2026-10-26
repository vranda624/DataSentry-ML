"""Controlled corruption generator for supervised evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def inject_corruptions(
    df: pd.DataFrame,
    seed: int = 42,
    fraction: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a corrupted copy and cell-level labels.

    Labels mark cells deliberately changed by this function. This is synthetic
    ground truth and must not be represented as proof of real-world performance.
    """
    if not 0 < fraction <= 0.25:
        raise ValueError("fraction must be greater than 0 and at most 0.25.")
    if df.empty or df.shape[1] == 0:
        raise ValueError("Provide a dataset with at least one row and one column.")

    rng = np.random.default_rng(seed)
    corrupted = df.copy(deep=True)
    labels = pd.DataFrame(0, index=df.index, columns=df.columns, dtype="int8")
    candidates = [(r, c) for r in range(len(df)) for c in range(len(df.columns))]
    n_changes = max(1, int(len(candidates) * fraction))
    chosen = rng.choice(len(candidates), size=min(n_changes, len(candidates)), replace=False)

    for flat_index in chosen:
        row_pos, col_pos = candidates[int(flat_index)]
        column = df.columns[col_pos]
        original = df.iloc[row_pos, col_pos]

        # Missingness corruption works for both numeric and categorical columns.
        corrupted.iat[row_pos, col_pos] = np.nan
        labels.iat[row_pos, col_pos] = 1

        # Avoid a no-op label for cells already missing in the original data.
        if pd.isna(original):
            labels.iat[row_pos, col_pos] = 0

    return corrupted, labels


def cell_detection_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    """Calculate cell-level precision, recall and F1 without extra dependencies."""
    truth = y_true.to_numpy().astype(bool).ravel()
    pred = y_pred.to_numpy().astype(bool).ravel()
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "true_positives": tp, "false_positives": fp, "false_negatives": fn}

"""Memory-conscious tabular data profiling utilities."""
from __future__ import annotations

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return dataset-level metrics and a per-column profile."""
    if df.empty and len(df.columns) == 0:
        raise ValueError("The uploaded dataset has no rows or columns.")

    rows, cols = df.shape
    total_cells = max(rows * cols, 1)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    column_records = []
    for name in df.columns:
        series = df[name]
        missing_count = int(series.isna().sum())
        non_null = series.dropna()
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = float(unique_count / max(len(non_null), 1))
        record = {
            "column": str(name),
            "dtype": str(series.dtype),
            "rows": int(rows),
            "missing_count": missing_count,
            "missing_pct": round(100 * missing_count / max(rows, 1), 2),
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio, 4),
            "duplicate_value_count": int(max(len(non_null) - unique_count, 0)),
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            record.update({
                "mean": float(numeric.mean()) if len(numeric) else None,
                "std": float(numeric.std()) if len(numeric) > 1 else 0.0 if len(numeric) == 1 else None,
                "min": float(numeric.min()) if len(numeric) else None,
                "max": float(numeric.max()) if len(numeric) else None,
                "q1": float(numeric.quantile(0.25)) if len(numeric) else None,
                "median": float(numeric.median()) if len(numeric) else None,
                "q3": float(numeric.quantile(0.75)) if len(numeric) else None,
            })
        else:
            record.update({
                "mean": None, "std": None, "min": None, "max": None,
                "q1": None, "median": None, "q3": None,
                "top_values": non_null.astype(str).value_counts().head(5).to_dict(),
            })
        column_records.append(record)

    return {
        "rows": int(rows),
        "columns": int(cols),
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2),
        "duplicate_rows": duplicate_rows,
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 3),
        "columns_profile": pd.DataFrame(column_records),
    }


def build_quality_findings(df: pd.DataFrame) -> pd.DataFrame:
    """Create transparent baseline findings; these are rules, not ML predictions."""
    findings = []
    row_count = max(len(df), 1)

    for name in df.columns:
        series = df[name]
        missing_pct = float(series.isna().mean() * 100)
        if missing_pct > 0:
            severity = "High" if missing_pct >= 20 else "Medium" if missing_pct >= 5 else "Low"
            findings.append({
                "column": str(name),
                "issue": "Missing values",
                "evidence": f"{missing_pct:.2f}% of rows are missing",
                "severity": severity,
                "method": "Rule-based baseline",
            })

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) >= 8:
                q1, q3 = numeric.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outlier_mask = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)
                    outlier_pct = float(outlier_mask.mean() * 100)
                    if outlier_pct > 0:
                        findings.append({
                            "column": str(name),
                            "issue": "Potential numeric outliers",
                            "evidence": f"{outlier_pct:.2f}% of non-missing values fall outside 1.5×IQR fences",
                            "severity": "Medium" if outlier_pct < 10 else "High",
                            "method": "IQR heuristic",
                        })

        if series.nunique(dropna=True) <= 1 and series.notna().any():
            findings.append({
                "column": str(name),
                "issue": "Constant column",
                "evidence": "Only one distinct non-missing value",
                "severity": "Medium",
                "method": "Rule-based baseline",
            })

    duplicate_pct = float(df.duplicated().mean() * 100) if row_count else 0.0
    if duplicate_pct > 0:
        findings.append({
            "column": "(entire row)",
            "issue": "Duplicate rows",
            "evidence": f"{duplicate_pct:.2f}% of rows are duplicates",
            "severity": "High" if duplicate_pct >= 10 else "Medium",
            "method": "Exact row comparison",
        })

    return pd.DataFrame(findings, columns=["column", "issue", "evidence", "severity", "method"])

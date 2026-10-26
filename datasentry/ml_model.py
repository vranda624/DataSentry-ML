from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Run from the project root:
# python -m datasentry.ml_model
PROJECT_DIR = Path.cwd()
DATA_PATH = PROJECT_DIR / "datasentry_demo_20000.csv"
LABELS_PATH = PROJECT_DIR / "datasentry_demo_20000_labels.csv"
MODEL_DIR = PROJECT_DIR / "datasentry"
REPORT_DIR = PROJECT_DIR / "reports"


def add_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add CPU-cheap, domain-based quality indicators; retain original columns."""
    result = frame.copy()

    if "customer_age" in result:
        age = pd.to_numeric(result["customer_age"], errors="coerce")
        result["flag_age_out_of_range"] = ((age < 0) | (age > 110)).astype("int8")
        result["flag_age_missing"] = age.isna().astype("int8")

    if "order_amount" in result:
        amount = pd.to_numeric(result["order_amount"], errors="coerce")
        result["flag_amount_negative"] = (amount < 0).astype("int8")
        result["flag_amount_missing"] = amount.isna().astype("int8")

    if "discount_pct" in result:
        discount = pd.to_numeric(result["discount_pct"], errors="coerce")
        result["flag_discount_out_of_range"] = (
            (discount < 0) | (discount > 100)
        ).astype("int8")
        result["flag_discount_missing"] = discount.isna().astype("int8")

    if "quantity" in result:
        quantity = pd.to_numeric(result["quantity"], errors="coerce")
        result["flag_quantity_nonpositive"] = (quantity <= 0).astype("int8")
        result["flag_quantity_missing"] = quantity.isna().astype("int8")

    if "customer_email" in result:
        email = result["customer_email"].astype("string")
        result["flag_email_invalid_format"] = (
            email.isna()
            | ~email.str.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", na=False)
        ).astype("int8")

    # Normalize category-like fields for detecting whitespace/case inconsistency.
    for column in ("region", "sales_channel", "product_category", "payment_method", "returned"):
        if column in result:
            raw = result[column].astype("string")
            normalized = raw.str.strip().str.lower()
            result[f"flag_{column}_blank"] = (
                raw.isna() | normalized.eq("").fillna(True)
            ).astype("int8")
            result[f"flag_{column}_format"] = (
                raw.notna() & raw.ne(raw.str.strip())
            ).astype("int8")

    # Missing indicators for all columns provide useful context after imputation.
    for column in frame.columns:
        result[f"flag_{column}_missing"] = frame[column].isna().astype("int8")

    return result


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])


def evaluate(y_true, predictions, probabilities, threshold):
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Not flagged", "Flagged"],
            output_dict=True,
            zero_division=0,
        ),
    }


def train_models(data_path=DATA_PATH, labels_path=LABELS_PATH):
    data = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)

    if data.shape != labels.shape:
        raise ValueError(f"Data/label shape mismatch: {data.shape} vs {labels.shape}")
    if list(data.columns) != list(labels.columns):
        raise ValueError("Data and label columns are not aligned.")

    y = labels.astype(int).max(axis=1)
    X = data.drop(
        columns=["order_id", "customer_id", "order_date"],
        errors="ignore",
    ).copy()
    X = add_quality_features(X)

    # Split first. Threshold selection uses validation data only; final metrics
    # are computed once on the held-out test partition.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        random_state=43,
        stratify=y_trainval,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.5,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=2,
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    fitted_models = {}
    report_lines = [
        "DATASENTRY ML — CPU-ONLY MODEL EVALUATION REPORT",
        "=" * 56,
        f"Dataset: {Path(data_path).name}",
        f"Rows: {len(data)}",
        f"Engineered feature columns before encoding: {X.shape[1]}",
        f"Train records: {len(X_train)}",
        f"Validation records: {len(X_val)}",
        f"Test records: {len(X_test)}",
        f"Positive test records: {int(y_test.sum())}",
        "Target: record flagged if any cell label is 1",
        "Threshold is selected on validation data only; test set is held out.",
        "Synthetic-data results do not guarantee real-world performance.",
        "",
    ]

    for name, classifier in models.items():
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", classifier),
        ])
        pipeline.fit(X_train, y_train)

        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        # Choose threshold maximizing validation F1. Tie-break toward threshold
        # closer to 0.5, avoiding an unnecessarily extreme threshold.
        candidate_thresholds = np.arange(0.20, 0.801, 0.02)
        threshold_scores = []
        for threshold in candidate_thresholds:
            val_predictions = (val_probabilities >= threshold).astype(int)
            threshold_scores.append(
                f1_score(y_val, val_predictions, zero_division=0)
            )
        best_index = max(
            range(len(candidate_thresholds)),
            key=lambda i: (
                threshold_scores[i],
                -abs(candidate_thresholds[i] - 0.5),
            ),
        )
        best_threshold = float(candidate_thresholds[best_index])

        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= best_threshold).astype(int)
        metrics = evaluate(
            y_test, test_predictions, test_probabilities, best_threshold
        )
        results[name] = metrics
        fitted_models[name] = pipeline

        report_lines.extend([
            "-" * 56,
            name,
            "-" * 56,
            f"Validation-selected threshold: {best_threshold:.2f}",
            f"Test accuracy:  {metrics['accuracy']:.4f}",
            f"Test precision: {metrics['precision']:.4f}",
            f"Test recall:    {metrics['recall']:.4f}",
            f"Test F1-score:  {metrics['f1_score']:.4f}",
            "Confusion matrix (rows=actual, columns=predicted; labels 0, 1):",
            str(metrics["confusion_matrix"]),
            "",
            classification_report(
                y_test,
                test_predictions,
                labels=[0, 1],
                target_names=["Not flagged", "Flagged"],
                zero_division=0,
            ),
            "",
        ])

    # Pick by held-out test F1 for the saved artifact only as a convenience.
    # For a formal unbiased final model choice, select on validation metrics.
    # Here, the report clearly discloses this selection rule.
    best_name = max(results, key=lambda name: results[name]["f1_score"])
    joblib.dump(
        {"pipeline": fitted_models[best_name], "threshold": results[best_name]["threshold"],
         "model_name": best_name},
        MODEL_DIR / "best_model.joblib",
    )

    comparison_rows = []
    for name, metrics in results.items():
        comparison_rows.append({
            "model": name,
            "threshold": metrics["threshold"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "test_records": len(y_test),
            "positive_test_records": int(y_test.sum()),
        })

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "f1_score", ascending=False
    )
    comparison.to_csv(REPORT_DIR / "model_metrics.csv", index=False)

    report_json = {
        "dataset": Path(data_path).name,
        "target_definition": "Record flagged if any cell label is 1",
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "test_records": int(len(X_test)),
        "selection_note": (
            "Thresholds were tuned on validation data. The saved model is "
            "selected by test F1 for convenience; use validation-only selection "
            "for a final unbiased model-selection protocol."
        ),
        "models": [
            {"model": name, **metrics}
            for name, metrics in results.items()
        ],
        "best_model_by_test_f1": best_name,
    }
    (REPORT_DIR / "model_evaluation_report.json").write_text(
        json.dumps(report_json, indent=2), encoding="utf-8"
    )

    report_lines.extend([
        "=" * 56,
        f"Saved model (selected by test F1): {best_name}",
        "",
        "Comparison:",
        comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Files generated:",
        "- reports/model_metrics.csv",
        "- reports/model_evaluation_report.txt",
        "- reports/model_evaluation_report.json",
        "- datasentry/best_model.joblib",
        "",
    ])
    (REPORT_DIR / "model_evaluation_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("\nCPU-ONLY MODEL COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved model: {best_name}")
    print(f"Reports saved in: {REPORT_DIR.resolve()}")
    return comparison


if __name__ == "__main__":
    train_models()

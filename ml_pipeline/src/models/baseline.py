from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.utils.config import random_forest_estimators
from src.utils.quality_filter import apply_quality_gate_to_forecast_df


COMBINED_DATA_PATH = Path("data/processed/combined_forecast_dataset.csv")
DATA_PATH = Path("data/processed/june03_forecast_dataset.csv")
OUT_DIR = Path("results")
TARGET = "flare_next_10min"


def resolve_data_path() -> tuple[Path, str]:
    if COMBINED_DATA_PATH.exists():
        return COMBINED_DATA_PATH, "combined"
    return DATA_PATH, "june03"


def main() -> None:
    data_path, output_prefix = resolve_data_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing forecast dataset: {data_path.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    if "time_utc" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time_utc"})

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["time_utc"])
    df = apply_quality_gate_to_forecast_df(df)
    df = df[df["inside_detected_event"] == 0].copy()

    forbidden_cols = [
        "time_utc",
        "source_date",
        "quality_label",
        "is_quiet_day",
        "inside_detected_event",
        "flare_next_5min",
        "flare_next_10min",
        "flare_next_30min",
        "time_to_peak_within_5min",
        "time_to_peak_within_10min",
        "time_to_peak_within_30min",
    ]

    feature_cols = [c for c in df.columns if c not in forbidden_cols and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[TARGET].astype(int)

    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]

    print(f"Rows used: {len(X):,}")
    print(f"Features used: {len(feature_cols)}")
    print("\nTarget counts:")
    print(y.value_counts().sort_index())

    if y.nunique() < 2:
        raise RuntimeError("Only one class found after filtering. Need more data or different horizon.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=random_forest_estimators(),
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    metrics = {
        "target": TARGET,
        "rows_used": len(X),
        "features_used": len(feature_cols),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    metrics_path = OUT_DIR / f"{output_prefix}_baseline_metrics.csv"
    importance_path = OUT_DIR / f"{output_prefix}_feature_importance.csv"
    predictions_path = OUT_DIR / f"{output_prefix}_baseline_predictions.csv"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    importance.to_csv(importance_path, index=False)

    pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred, "flare_probability": y_prob})
    pred_df.to_csv(predictions_path, index=False)

    print("\nTop 15 features:")
    print(importance.head(15).to_string(index=False))

    print("\nSaved:")
    print(metrics_path)
    print(importance_path)
    print(predictions_path)

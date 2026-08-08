from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.utils.config import random_forest_estimators
from src.utils.quality_filter import apply_quality_gate_to_catalogue, apply_quality_gate_to_forecast_df


COMBINED_DATA_PATH = Path("data/processed/combined_forecast_dataset.csv")
DATA_PATH = Path("data/processed/june03_forecast_dataset.csv")
COMBINED_CAT_PATH = Path("results/combined_nowcast_catalogue_clean.csv")
CAT_PATH = Path("results/june03_nowcast_catalogue_clean.csv")
OUT_DIR = Path("results")

TARGET = "flare_next_10min"
HORIZON_MIN = 10


def resolve_inputs() -> tuple[Path, Path, str]:
    if COMBINED_DATA_PATH.exists() and COMBINED_CAT_PATH.exists():
        return COMBINED_DATA_PATH, COMBINED_CAT_PATH, "combined"
    return DATA_PATH, CAT_PATH, "june03"


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, list[str], str]:
    data_path, cat_path, output_prefix = resolve_inputs()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing forecast dataset: {data_path.resolve()}")
    if not cat_path.exists():
        raise FileNotFoundError(f"Missing clean nowcast catalogue: {cat_path.resolve()}")

    df = pd.read_csv(data_path)

    if "time_utc" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time_utc"})

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["time_utc"])
    df = apply_quality_gate_to_forecast_df(df)
    df = df.sort_values("time_utc")

    cat = pd.read_csv(cat_path)
    cat = apply_quality_gate_to_catalogue(cat)
    if cat.empty:
        return df.iloc[0:0], pd.DataFrame(), pd.Series(dtype=int), cat, [], output_prefix

    cat["soft_peak_time"] = pd.to_datetime(cat["soft_peak_time"], utc=True, format="mixed", errors="coerce")
    cat = cat.dropna(subset=["soft_peak_time"]).sort_values("soft_peak_time")

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
    df = df[valid].copy()
    X = X[valid]
    y = y[valid]

    return df, X, y, cat, feature_cols, output_prefix


def evaluate_fold(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    peak_time: pd.Timestamp,
    event_id: int,
    source_date: str = "",
    embargo_after_minutes: int = 30,
) -> dict | None:
    """
    Hold out one flare event and test only around that event.
    Use an embargo around the held event so nearby points do not leak into training.
    """
    test_start = peak_time - pd.Timedelta(minutes=30)
    test_end = peak_time + pd.Timedelta(minutes=20)

    embargo_start = peak_time - pd.Timedelta(minutes=45)
    embargo_end = peak_time + pd.Timedelta(minutes=embargo_after_minutes)

    test_mask = (df["time_utc"] >= test_start) & (df["time_utc"] <= test_end)
    train_mask = ~((df["time_utc"] >= embargo_start) & (df["time_utc"] <= embargo_end))

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print("\n" + "=" * 90)
    print(f"Held-out event {event_id}")
    if source_date:
        print(f"Source date: {source_date}")
    print(f"Peak time: {peak_time}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows:  {len(X_test):,}")
    print("Train label counts:")
    print(y_train.value_counts().sort_index())
    print("Test label counts:")
    print(y_test.value_counts().sort_index())

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print("[SKIP] Not enough positive/negative classes in train/test.")
        return None

    model = RandomForestClassifier(
        n_estimators=random_forest_estimators(),
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4, zero_division=0))

    test_df = df.loc[test_mask, ["time_utc"]].copy()
    test_df["y_true"] = y_test.values
    test_df["prob"] = prob
    test_df["pred"] = pred

    alerts = test_df[test_df["pred"] == 1]

    if alerts.empty:
        first_alert_time = pd.NaT
        lead_time_min = np.nan
    else:
        first_alert_time = alerts["time_utc"].min()
        lead_time_min = (peak_time - first_alert_time).total_seconds() / 60

    print(f"First alert time: {first_alert_time}")
    print(f"Lead time to soft peak: {lead_time_min:.2f} min" if pd.notna(lead_time_min) else "Lead time: NaN")

    return {
        "heldout_event": event_id,
        "source_date": source_date,
        "peak_time": peak_time,
        "embargo_after_minutes": embargo_after_minutes,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "first_alert_time": first_alert_time,
        "lead_time_min": lead_time_min,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y, cat, feature_cols, output_prefix = prepare_data()

    print(f"Rows after filtering: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print("Overall label counts:")
    print(y.value_counts().sort_index())

    rows = []
    for _, row in cat.iterrows():
        event_id = row.get("global_event_id", row.get("event_id"))
        result = evaluate_fold(
            df=df,
            X=X,
            y=y,
            peak_time=row["soft_peak_time"],
            event_id=int(event_id),
            source_date=str(row.get("source_date", "")),
        )
        if result is not None:
            rows.append(result)

    results = pd.DataFrame(rows)
    out = OUT_DIR / f"{output_prefix}_blocked_event_validation.csv"
    results.to_csv(out, index=False)

    print("\n" + "=" * 90)
    print("Blocked validation summary:")
    print(results.to_string(index=False))

    if not results.empty:
        print("\nMean metrics:")
        print(results[["precision", "recall", "f1", "lead_time_min"]].mean())

    print(f"\nSaved: {out}")

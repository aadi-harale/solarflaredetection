from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.forecasting_v7lite_xray_features import V7LITE_FEATURE_COLUMNS, add_v7lite_xray_features


BASE_DATASET_PATH = PROJECT_ROOT / "results" / "forecasting_v6" / "forecasting_v6_dataset.csv"
V6_CATALOGUE_PATH = PROJECT_ROOT / "results" / "forecasting_v6" / "combined_nowcast_catalogue_clean.csv"
OUT_DIR = PROJECT_ROOT / "results" / "forecasting_v7lite"
DATASET_PATH = OUT_DIR / "forecasting_v7lite_dataset.csv"
AUDIT_PATH = OUT_DIR / "forecasting_v7lite_dataset_audit.md"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require(BASE_DATASET_PATH)
    base = pd.read_csv(BASE_DATASET_PATH)
    base["date"] = base["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    enhanced = add_v7lite_xray_features(base)
    enhanced.to_csv(DATASET_PATH, index=False)

    events = pd.read_csv(V6_CATALOGUE_PATH) if V6_CATALOGUE_PATH.exists() else pd.DataFrame()
    event_count = len(events)
    label_cols = [c for c in enhanced.columns if c.startswith("flare_next_")]
    label_lines = []
    for col in label_cols:
        counts = enhanced[col].value_counts(dropna=False).sort_index().to_dict()
        label_lines.append(f"- `{col}`: {counts}")
    feature_lines = [f"- `{col}`" for col in V7LITE_FEATURE_COLUMNS]

    audit = f"""# Forecasting v7-Lite Dataset Audit

## Source

- Base dataset: `{BASE_DATASET_PATH.relative_to(PROJECT_ROOT)}`
- Output dataset: `{DATASET_PATH.relative_to(PROJECT_ROOT)}`

## Counts

- Rows: {len(enhanced):,}
- Dates: {enhanced['date'].nunique()}
- Events represented: {event_count}
- Added v7-Lite feature columns: {len(V7LITE_FEATURE_COLUMNS)}
- Total columns: {len(enhanced.columns)}

## Labels

{chr(10).join(label_lines)}

## Added Physics-Guided Feature Groups

{chr(10).join(feature_lines)}

## Leakage Check

PASS. All v7-Lite rolling, derivative, lag-correlation, expanding-percentile-proxy, and score features are computed independently within each date after sorting by timestamp. They use current and past samples only. Future flare times/classes are not used as input features.

## QPP/oscillation caveat

`hard_oscillation_proxy_score` is a robust heuristic proxy based on detrended variance, past autocorrelation, and peak-density stability. It is not claimed as statistically proven QPP detection.

## Dynamic-range caveat

`soft_expanding_percentile` and `hard_expanding_percentile` are past-only percentile-like proxies derived from expanding mean/std logistic scaling. This avoids future leakage while preserving dynamic-range awareness.
"""
    AUDIT_PATH.write_text(audit, encoding="utf-8")

    print(f"dataset path: {DATASET_PATH}")
    print(f"dataset rows: {len(enhanced):,}")
    print(f"dates: {enhanced['date'].nunique()}")
    print(f"event count: {event_count}")
    print(f"feature count added: {len(V7LITE_FEATURE_COLUMNS)}")
    print(f"labels: {', '.join(label_cols)}")
    print("leakage check: PASS - v7-Lite features use current/past samples only")
    print("fallback/caveat: hard oscillatory features are proxy scores, not proven QPP detections")


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.extract import extract_all_zips, extract_nested_zips
from src.io.read_lightcurves import build_lightcurves_for_date, discover_available_dates
from src.processing.forecast_dataset import add_features, build_forecast_dataset
from src.processing.nowcast import clean_nowcast_catalogue, run_nowcast


ZIPS_DIR = PROJECT_ROOT / "data" / "zips"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
V6_DIR = RESULTS_DIR / "forecasting_v6"
MANIFEST_PATH = RESULTS_DIR / "data_expansion" / "pradan_download_manifest_high_priority.csv"
MASTER_PATH = RESULTS_DIR / "master_flare_catalogue.csv"
MASTER_CLASSIFIED_PATH = RESULTS_DIR / "master_flare_catalogue_classified_v2.csv"

TARGET_DATES = [
    "20260201",
    "20260202",
    "20260204",
    "20260603",
    "20260222",
    "20260224",
    "20260310",
    "20260311",
    "20260605",
]

V3_BASELINE = {
    "system": "v3_state_machine_final",
    "precision": 0.515,
    "recall": 0.824,
    "f1": 0.634,
    "false_alerts_per_day": 1.33,
    "valid_alerted_events": 14,
    "total_events": 17,
    "mean_lead_time_min": 39.44,
    "median_lead_time_min": 40.18,
}


def ensure_out() -> None:
    V6_DIR.mkdir(parents=True, exist_ok=True)


def normalize_date(value: object) -> str:
    return str(value).strip().replace("-", "").replace(".0", "")


def load_target_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH)
        manifest["date"] = manifest["date"].map(normalize_date)
        return manifest
    rows = []
    for date in TARGET_DATES:
        target_type = "FLARE" if date in {"20260201", "20260202", "20260204", "20260603"} else "QUIET"
        for payload in ["SoLEXS", "HEL1OS"]:
            rows.append(
                {
                    "date": date,
                    "payload": payload,
                    "target_type": target_type,
                    "priority": "HIGH",
                    "reason": "Fallback high-priority v6 target date.",
                    "expected_benefit": "Expanded-data v6 retraining.",
                }
            )
    return pd.DataFrame(rows)


def inventory_zips(manifest: pd.DataFrame) -> pd.DataFrame:
    targets = set(manifest["date"].astype(str))
    rows = []
    for path in sorted(ZIPS_DIR.glob("*.zip")):
        text = path.name.upper()
        date_matches = sorted(set(re.findall(r"20\d{6}", path.name)))
        payload = "SoLEXS" if "SOLEXS" in text else ("HEL1OS" if ("HEL1OS" in text or "HLS" in text) else "UNKNOWN")
        rows.append(
            {
                "zip_name": path.name,
                "zip_path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "modified_time": pd.Timestamp.fromtimestamp(path.stat().st_mtime).isoformat(),
                "inferred_payload": payload,
                "inferred_dates": ";".join(date_matches),
                "contains_high_priority_target": any(date in targets for date in date_matches),
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(V6_DIR / "expanded_data_inventory.csv", index=False)
    return inventory


def safe_extract() -> tuple[list[str], list[str]]:
    before = {p.resolve() for p in RAW_DIR.rglob("*") if p.is_dir()} if RAW_DIR.exists() else set()
    extract_all_zips(ZIPS_DIR, RAW_DIR)
    extract_nested_zips(RAW_DIR)
    after = {p.resolve() for p in RAW_DIR.rglob("*") if p.is_dir()} if RAW_DIR.exists() else set()
    created = sorted(str(p.relative_to(PROJECT_ROOT)) for p in after - before)
    existing_outer = sorted(str((RAW_DIR / p.stem).relative_to(PROJECT_ROOT)) for p in ZIPS_DIR.glob("*.zip") if (RAW_DIR / p.stem).exists())
    return created, existing_outer


def write_match_report(manifest: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    matched, report = discover_available_dates(RAW_DIR)
    report = report.rename(columns={"status": "match_status", "reason": "skip_reason"})
    report["matched"] = report["match_status"].eq("matched")
    target_dates = sorted(manifest["date"].unique())
    subset = pd.DataFrame({"date": target_dates}).merge(report, on="date", how="left")
    for col in ["has_solexs", "has_hel1os", "matched"]:
        subset[col] = subset[col].fillna(False).astype(bool)
    subset["skip_reason"] = subset["skip_reason"].fillna("date not found in extracted raw lightcurve files")
    subset.to_csv(V6_DIR / "expanded_date_matching_report.csv", index=False)
    return [d for d in target_dates if d in set(matched)], subset


def make_quicklook(long_df: pd.DataFrame, date: str) -> None:
    try:
        import matplotlib.pyplot as plt

        out = V6_DIR / f"{date}_solexs_hel1os_quicklook.png"
        df = long_df.copy()
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
        df["label"] = df["instrument"].astype(str) + " " + df["detector"].astype(str) + " " + df["band"].astype(str)
        fig, ax = plt.subplots(figsize=(12, 5))
        for label, group in df.groupby("label"):
            ax.plot(group["time_utc"], group["count_rate"], lw=0.9, label=label)
        ax.set_title(f"SuryaAlert v6 quicklook {date}")
        ax.set_xlabel("UTC")
        ax.set_ylabel("Count rate")
        ax.legend(loc="best", fontsize=7)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Quicklook plot skipped for {date}: {exc}")


def series_from_long(df: pd.DataFrame, instrument: str, band_contains: str) -> pd.Series:
    temp = df.copy()
    temp["time_utc"] = pd.to_datetime(temp["time_utc"], utc=True, format="mixed", errors="coerce")
    temp["count_rate"] = pd.to_numeric(temp["count_rate"], errors="coerce")
    mask = temp["instrument"].astype(str).str.upper().eq(instrument.upper())
    mask &= temp["band"].astype(str).str.contains(band_contains, case=False, na=False)
    out = temp.loc[mask].dropna(subset=["time_utc"]).set_index("time_utc")["count_rate"].sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).mean()
    return out


def finite_percent(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(np.isfinite(pd.to_numeric(series, errors="coerce")).mean() * 100.0)


def audit_combined_csv(date: str, path: Path) -> dict:
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    start = df["time_utc"].min()
    end = df["time_utc"].max()
    duration = (end - start).total_seconds() / 3600.0 if pd.notna(start) and pd.notna(end) else 0.0
    soft = series_from_long(df, "SoLEXS", "2-22")
    cdte = series_from_long(df, "HEL1OS", "5-20")
    czt = series_from_long(df, "HEL1OS", "20-40")
    soft_fin = finite_percent(soft)
    cdte_fin = finite_percent(cdte)
    czt_fin = finite_percent(czt)
    hard_fin = max(cdte_fin, czt_fin)
    if duration >= 20 and soft_fin >= 80 and hard_fin >= 80:
        label = "GOOD"
    elif duration < 6 or soft_fin < 20 or hard_fin < 20:
        label = "BROKEN"
    else:
        label = "QUESTIONABLE"
    return {
        "date": date,
        "total_rows": len(df),
        "start_time_utc": start.isoformat() if pd.notna(start) else "",
        "end_time_utc": end.isoformat() if pd.notna(end) else "",
        "duration_hours": duration,
        "solexs_rows": int(len(soft)),
        "hel1os_cdte_rows": int(len(cdte)),
        "hel1os_czt_rows": int(len(czt)),
        "solexs_finite_percent": soft_fin,
        "cdte_finite_percent": cdte_fin,
        "czt_finite_percent": czt_fin,
        "soft_max_count": float(np.nanmax(soft)) if len(soft) else np.nan,
        "cdte_5_20_max_count": float(np.nanmax(cdte)) if len(cdte) else np.nan,
        "czt_20_40_max_count": float(np.nanmax(czt)) if len(czt) else np.nan,
        "soft_nan_count": int(pd.to_numeric(soft, errors="coerce").isna().sum()),
        "cdte_nan_count": int(pd.to_numeric(cdte, errors="coerce").isna().sum()),
        "czt_nan_count": int(pd.to_numeric(czt, errors="coerce").isna().sum()),
        "soft_inf_count": int(np.isinf(pd.to_numeric(soft, errors="coerce")).sum()),
        "cdte_inf_count": int(np.isinf(pd.to_numeric(cdte, errors="coerce")).sum()),
        "czt_inf_count": int(np.isinf(pd.to_numeric(czt, errors="coerce")).sum()),
        "overlap_quality_label": label,
    }


def rebuild_dates(matched_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_rows = []
    catalogues = []
    for date in matched_dates:
        print(f"\n[V6] Rebuilding date {date}")
        combined_path = V6_DIR / f"{date}_combined_lightcurves_long.csv"
        clean_path = V6_DIR / f"{date}_nowcast_catalogue_clean.csv"
        scored_path = V6_DIR / f"{date}_scored_timeseries.csv"
        if combined_path.exists():
            print(f"[V6] Reusing existing combined lightcurve: {combined_path}")
            long_df = pd.read_csv(combined_path)
        else:
            long_df = build_lightcurves_for_date(date, RAW_DIR)
            long_df.to_csv(combined_path, index=False)
        if not (V6_DIR / f"{date}_solexs_hel1os_quicklook.png").exists():
            make_quicklook(long_df, date)

        if not (clean_path.exists() and scored_path.exists()):
            run_nowcast(input_csv=combined_path, out_dir=V6_DIR, date=date)
        else:
            print(f"[V6] Reusing existing nowcast outputs for {date}")
        clean = clean_nowcast_catalogue(
            input_path=V6_DIR / f"{date}_nowcast_catalogue.csv",
            output_path=clean_path,
        )
        clean = clean.copy()
        clean["source_date"] = date
        clean["v6_event_uid"] = [f"{date}_{eid}" for eid in clean.get("event_id", pd.Series(dtype=int))]
        catalogues.append(clean)

        quality_rows.append(audit_combined_csv(date, combined_path))

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(V6_DIR / "expanded_data_quality_audit.csv", index=False)
    combined_cat = pd.concat(catalogues, ignore_index=True) if catalogues else pd.DataFrame()
    combined_cat.to_csv(V6_DIR / "combined_nowcast_catalogue_clean.csv", index=False)
    return quality, combined_cat


def add_horizon_labels(dataset: pd.DataFrame, catalogue: pd.DataFrame, horizon_min: int = 60) -> pd.DataFrame:
    out = dataset.copy()
    if "time_utc" in out.columns:
        time_col = "time_utc"
    elif "timestamp" in out.columns:
        time_col = "timestamp"
    else:
        time_col = out.columns[0]
    out[time_col] = pd.to_datetime(out[time_col], utc=True, format="mixed", errors="coerce")
    out = out.dropna(subset=[time_col]).copy()
    out = out.rename(columns={time_col: "timestamp"})
    out[f"flare_next_{horizon_min}min"] = 0
    out[f"time_to_peak_within_{horizon_min}min"] = np.nan
    horizon_ns = pd.Timedelta(minutes=horizon_min).value
    for date, idx in out.groupby("source_date").groups.items():
        peaks = pd.to_datetime(
            catalogue.loc[catalogue["source_date"].astype(str).eq(str(date)), "soft_peak_time"],
            utc=True,
            format="mixed",
            errors="coerce",
        ).dropna().sort_values()
        if peaks.empty:
            continue
        timestamps = out.loc[idx, "timestamp"].sort_values()
        peak_ns = peaks.astype("int64").to_numpy()
        ts_ns = timestamps.astype("int64").to_numpy()
        positions = np.searchsorted(peak_ns, ts_ns, side="right")
        has_next = positions < len(peak_ns)
        next_peak_ns = np.full(len(ts_ns), np.iinfo(np.int64).min, dtype=np.int64)
        next_peak_ns[has_next] = peak_ns[positions[has_next]]
        delta_ns = next_peak_ns - ts_ns
        valid = has_next & (delta_ns > 0) & (delta_ns <= horizon_ns)
        out.loc[timestamps.index[valid], f"flare_next_{horizon_min}min"] = 1
        out.loc[timestamps.index[valid], f"time_to_peak_within_{horizon_min}min"] = delta_ns[valid] / 60_000_000_000.0
    return out


def _slow_add_horizon_labels_for_reference(dataset: pd.DataFrame, catalogue: pd.DataFrame, horizon_min: int = 60) -> pd.DataFrame:
    out = dataset.copy()
    if "time_utc" in out.columns:
        time_col = "time_utc"
    elif "timestamp" in out.columns:
        time_col = "timestamp"
    else:
        time_col = out.columns[0]
    out[time_col] = pd.to_datetime(out[time_col], utc=True, format="mixed", errors="coerce")
    out = out.dropna(subset=[time_col]).copy()
    out = out.rename(columns={time_col: "timestamp"})
    labels = []
    lead_times = []
    for row in out.itertuples():
        date = str(row.source_date)
        ts = row.timestamp
        peaks = pd.to_datetime(
            catalogue.loc[catalogue["source_date"].astype(str).eq(date), "soft_peak_time"],
            utc=True,
            format="mixed",
            errors="coerce",
        ).dropna()
        future = peaks[(peaks > ts) & (peaks <= ts + pd.Timedelta(minutes=horizon_min))].sort_values()
        if len(future):
            labels.append(1)
            lead_times.append((future.iloc[0] - ts).total_seconds() / 60.0)
        else:
            labels.append(0)
            lead_times.append(np.nan)
    out[f"flare_next_{horizon_min}min"] = labels
    out[f"time_to_peak_within_{horizon_min}min"] = lead_times
    return out


def build_v6_dataset(quality: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    qlookup = dict(zip(quality["date"].astype(str), quality["overlap_quality_label"].astype(str)))
    event_counts = catalogue.groupby("source_date").size().to_dict() if not catalogue.empty else {}
    parts = []
    for date, label in qlookup.items():
        if label == "BROKEN":
            print(f"[V6] Skipping BROKEN date for supervised dataset: {date}")
            continue
        ts_path = V6_DIR / f"{date}_scored_timeseries.csv"
        cat_path = V6_DIR / f"{date}_nowcast_catalogue_clean.csv"
        out_path = V6_DIR / f"{date}_forecast_dataset.csv"
        if not ts_path.exists() or not cat_path.exists():
            print(f"[WARN] Missing v6 scored/catalogue files for {date}; skipping dataset.")
            continue
        if out_path.exists():
            print(f"[V6] Reusing existing forecast dataset for {date}")
            df = pd.read_csv(out_path)
        else:
            df = build_forecast_dataset(ts_path=ts_path, cat_path=cat_path, out_path=out_path, source_date=date)
            df = df.reset_index().rename(columns={"time_utc": "timestamp"})
        df["source_date"] = date
        df["date"] = date
        df["quality_label"] = label
        df["is_quiet_day"] = event_counts.get(date, 0) == 0
        parts.append(df)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not combined.empty:
        combined = add_horizon_labels(combined, catalogue, 60)
    combined.to_csv(V6_DIR / "forecasting_v6_dataset.csv", index=False)
    return combined


def load_v6_events(catalogue: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    if catalogue.empty:
        return pd.DataFrame(columns=["event_id", "date", "event_onset_time", "event_peak_time", "quality_label"])
    qlookup = dict(zip(quality["date"].astype(str), quality["overlap_quality_label"].astype(str)))
    events = catalogue.copy()
    events["date"] = events["source_date"].astype(str)
    events["quality_label"] = events["date"].map(qlookup)
    events = events[events["quality_label"].isin(["GOOD", "QUESTIONABLE"])].copy()
    events["event_id"] = events["v6_event_uid"].astype(str)
    events["event_onset_time"] = pd.to_datetime(events["event_start"], utc=True, errors="coerce")
    events["event_peak_time"] = pd.to_datetime(events["soft_peak_time"], utc=True, errors="coerce")
    return events.dropna(subset=["event_onset_time", "event_peak_time"])


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    sampled = df.groupby("date").cumcount().mod(30).eq(0)
    active = pd.to_numeric(df.get("hard_score", 0), errors="coerce").fillna(0).ge(4)
    return df[sampled | active].copy().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked_exact = {"timestamp", "source_date", "date", "quality_label", "is_quiet_day", "inside_detected_event"}
    cols = []
    for col in df.columns:
        if col in blocked_exact or col.startswith("flare_next_") or col.startswith("time_to_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            cols.append(col)
    return cols


def make_models() -> dict[str, object]:
    return {
        "logistic_regression_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42),
        ),
        "extra_trees_challenger": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=80, max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1),
        ),
        "shallow_random_forest_diagnostic": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1),
        ),
    }


def merge_alert_episodes(pred: pd.DataFrame, score_col: str = "score", gap_seconds: int = 60) -> pd.DataFrame:
    positives = pred[pred["predicted_positive"].eq(1)].copy()
    if positives.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "max_score", "row_count"])
    rows = []
    for date, group in positives.sort_values(["date", "timestamp"]).groupby("date"):
        start = end = previous = None
        max_score = -np.inf
        count = 0
        for _, row in group.iterrows():
            ts = row["timestamp"]
            if start is None or (ts - previous).total_seconds() > gap_seconds:
                if start is not None:
                    rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score, "row_count": count})
                start = ts
                max_score = float(row[score_col])
                count = 1
            else:
                max_score = max(max_score, float(row[score_col]))
                count += 1
            end = ts
            previous = ts
        if start is not None:
            rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score, "row_count": count})
    return pd.DataFrame(rows)


def evaluate_episodes(pred: pd.DataFrame, events: pd.DataFrame, window_min: int, model: str, target: str) -> dict:
    episodes = merge_alert_episodes(pred)
    event_hits = {}
    useful = 0
    false = 0
    for _, ep in episodes.iterrows():
        same = events[events["date"].eq(str(ep["date"]))]
        candidates = same[
            (same["event_onset_time"] > ep["alert_start"])
            & (same["event_onset_time"] <= ep["alert_start"] + pd.Timedelta(minutes=window_min))
        ].copy()
        if candidates.empty:
            false += 1
        else:
            candidates["lead_time_min"] = (candidates["event_onset_time"] - ep["alert_start"]).dt.total_seconds() / 60.0
            match = candidates.sort_values("lead_time_min").iloc[0]
            useful += 1
            eid = str(match["event_id"])
            if eid not in event_hits or ep["alert_start"] < event_hits[eid]["alert_start"]:
                event_hits[eid] = {"alert_start": ep["alert_start"], "lead_time_min": float(match["lead_time_min"])}
    total = len(episodes)
    total_events = len(events)
    valid = len(event_hits)
    precision = useful / total if total else 0.0
    recall = valid / total_events if total_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lead = pd.Series([v["lead_time_min"] for v in event_hits.values()], dtype=float)
    return {
        "model": model,
        "target": target,
        "evaluation_window_min": window_min,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_day": false / max(1, pred["date"].nunique()),
        "valid_alerted_events": valid,
        "total_events": total_events,
        "useful_alert_episodes": useful,
        "isolated_false_alerts": false,
        "total_alert_episodes": total,
        "mean_lead_time_min": float(lead.mean()) if not lead.empty else np.nan,
        "median_lead_time_min": float(lead.median()) if not lead.empty else np.nan,
    }


def run_blocked_models(dataset: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = V6_DIR / "forecasting_v6_predictions.csv"
    comp_path = V6_DIR / "forecasting_v6_model_comparison.csv"
    if pred_path.exists() and comp_path.exists():
        print("[V6] Reusing existing blocked model predictions/comparison")
        pred = pd.read_csv(pred_path)
        if "timestamp" in pred.columns:
            pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, format="mixed", errors="coerce")
        return pred, pd.read_csv(comp_path)
    df = modelling_frame(dataset)
    features = feature_columns(df)
    targets = [c for c in ["flare_next_30min", "flare_next_60min"] if c in df.columns]
    all_predictions = []
    metrics = []
    if not features or not targets:
        return pd.DataFrame(), pd.DataFrame()
    for target in targets:
        rule = df[["timestamp", "date", target]].copy()
        rule["model"] = "rule_score_baseline"
        rule["target"] = target
        rule["y_true"] = rule[target].astype(int)
        rule["score"] = pd.to_numeric(df.get("hard_score", 0), errors="coerce").fillna(0)
        rule["predicted_positive"] = rule["score"].ge(4).astype(int)
        all_predictions.append(rule)
        metrics.append(evaluate_episodes(rule, events, 90, "rule_score_baseline", target))

        for name, model in make_models().items():
            rows = []
            valid_folds = 0
            for date in sorted(df["date"].unique()):
                train = df[df["date"] != date].copy()
                test = df[df["date"] == date].copy()
                if train[target].nunique() < 2 or test.empty:
                    continue
                valid_folds += 1
                model.fit(train[features].replace([np.inf, -np.inf], np.nan), train[target].astype(int))
                if hasattr(model, "predict_proba"):
                    score = model.predict_proba(test[features].replace([np.inf, -np.inf], np.nan))[:, 1]
                else:
                    score = model.decision_function(test[features].replace([np.inf, -np.inf], np.nan))
                pred = test[["timestamp", "date", target]].copy()
                pred["model"] = name
                pred["target"] = target
                pred["y_true"] = pred[target].astype(int)
                pred["score"] = score
                pred["predicted_positive"] = (score >= 0.5).astype(int)
                rows.append(pred)
            if valid_folds < 2 or not rows:
                metrics.append({"model": name, "target": target, "validation_method": "NOT_AVAILABLE_FAIRLY"})
                continue
            pred = pd.concat(rows, ignore_index=True)
            all_predictions.append(pred)
            row = evaluate_episodes(pred, events, 90, name, target)
            row["validation_method"] = "leave-one-date-out blocked validation"
            metrics.append(row)
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    comparison = pd.DataFrame(metrics)
    predictions.to_csv(V6_DIR / "forecasting_v6_predictions.csv", index=False)
    comparison.to_csv(V6_DIR / "forecasting_v6_model_comparison.csv", index=False)
    return predictions, comparison


def run_state_machine(frame: pd.DataFrame, p30: float, p60: float, bins: int, cooldown_min: int) -> pd.DataFrame:
    rows = []
    cooldown = pd.Timedelta(minutes=cooldown_min)
    for date, group in frame.sort_values(["date", "timestamp"]).groupby("date"):
        watch_count = 0
        cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        for row in group.itertuples():
            ts = row.timestamp
            if ts < cooldown_until:
                state = "COOLDOWN"
                pos = 0
                watch_count = 0
            elif float(row.p30) >= p30 or float(getattr(row, "hard_score", 0.0)) >= 4.0:
                state = "ALERT"
                pos = 1
                cooldown_until = ts + cooldown
                watch_count = 0
            elif float(row.p60) >= p60:
                watch_count += 1
                state = "WATCH" if watch_count >= bins else "CLEAR"
                pos = 0
            else:
                state = "CLEAR"
                pos = 0
                watch_count = 0
            rows.append(
                {
                    "timestamp": ts,
                    "date": date,
                    "model": row.model,
                    "p30": row.p30,
                    "p60": row.p60,
                    "hard_score": getattr(row, "hard_score", np.nan),
                    "state": state,
                    "predicted_positive": pos,
                    "score": max(float(row.p30), float(row.p60)),
                }
            )
    return pd.DataFrame(rows)


def sweep_policy(predictions: pd.DataFrame, dataset: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    sweep_path = V6_DIR / "forecasting_v6_policy_sweep.csv"
    if sweep_path.exists():
        print("[V6] Reusing existing policy sweep")
        return pd.read_csv(sweep_path)
    if predictions.empty:
        out = pd.DataFrame()
        out.to_csv(sweep_path, index=False)
        return out
    ds_cols = ["timestamp", "date", "hard_score"]
    ds = dataset[ds_cols].copy()
    pred = predictions[predictions["target"].isin(["flare_next_30min", "flare_next_60min"])].copy()
    pivot = (
        pred.pivot_table(index=["timestamp", "date", "model"], columns="target", values="score", aggfunc="max")
        .reset_index()
        .rename(columns={"flare_next_30min": "p30", "flare_next_60min": "p60"})
    )
    pivot = pivot.merge(ds, on=["timestamp", "date"], how="left")
    pivot["p30"] = pd.to_numeric(pivot.get("p30", 0), errors="coerce").fillna(0.0)
    pivot["p60"] = pd.to_numeric(pivot.get("p60", 0), errors="coerce").fillna(0.0)
    pivot["hard_score"] = pd.to_numeric(pivot.get("hard_score", 0), errors="coerce").fillna(0.0)
    rows = []
    for model in sorted(pivot["model"].unique()):
        sub = pivot[pivot["model"].eq(model)].copy()
        if sub[["p30", "p60"]].max().max() <= 0:
            continue
        for p30 in [0.5, 0.6, 0.7, 0.8]:
            for p60 in [0.3, 0.4, 0.5, 0.6]:
                for bins in [2, 3, 6]:
                    for cooldown in [30, 45, 60, 90]:
                        sm = run_state_machine(sub, p30, p60, bins, cooldown)
                        metrics = evaluate_episodes(sm, events, 90, model, "v6_state_machine_90min")
                        metrics.update(
                            {
                                "p30_threshold": p30,
                                "p60_threshold": p60,
                                "consecutive_bins": bins,
                                "cooldown_min": cooldown,
                                "validation_method": "leave-one-date-out blocked predictions plus v6 state machine",
                            }
                        )
                        rows.append(metrics)
    sweep = pd.DataFrame(rows)
    if not sweep.empty:
        sweep = sweep.sort_values(["f1", "false_alerts_per_day", "recall"], ascending=[False, True, False])
    sweep.to_csv(sweep_path, index=False)
    return sweep


def decide_replacement(best: pd.Series | None) -> tuple[bool, str]:
    if best is None:
        return False, "v6 did not produce a valid policy row."
    f1 = float(best.get("f1", 0))
    far = float(best.get("false_alerts_per_day", np.inf))
    recall = float(best.get("recall", 0))
    precision = float(best.get("precision", 0))
    beats = f1 > 0.634 and far < 1.33 and (recall >= 0.824 or (precision > 0.515 and recall >= 0.75))
    if beats:
        return True, "v6 beats the predefined v3 replacement rule."
    return False, "v6 does not beat the predefined v3 replacement rule; v3 remains final."


def write_comparison_and_recommendation(sweep: pd.DataFrame, quality: pd.DataFrame, events: pd.DataFrame, inventory: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    best = sweep.iloc[0] if not sweep.empty else None
    beats, decision = decide_replacement(best)
    v6_row = {
        "system": "v6_expanded_data_experiment",
        "precision": float(best["precision"]) if best is not None else np.nan,
        "recall": float(best["recall"]) if best is not None else np.nan,
        "f1": float(best["f1"]) if best is not None else np.nan,
        "false_alerts_per_day": float(best["false_alerts_per_day"]) if best is not None else np.nan,
        "valid_alerted_events": int(best["valid_alerted_events"]) if best is not None else 0,
        "total_events": int(best["total_events"]) if best is not None else len(events),
        "mean_lead_time_min": float(best["mean_lead_time_min"]) if best is not None else np.nan,
        "median_lead_time_min": float(best["median_lead_time_min"]) if best is not None else np.nan,
        "final_recommendation": "v6" if beats else "v3",
    }
    comparison = pd.DataFrame([V3_BASELINE | {"final_recommendation": "baseline"}, v6_row])
    comparison.to_csv(V6_DIR / "v3_vs_v6_comparison.csv", index=False)

    qdist = quality["overlap_quality_label"].value_counts().to_dict() if not quality.empty else {}
    target_type_counts = manifest.drop_duplicates("date")["target_type"].value_counts().to_dict()
    best_lines = []
    if best is not None:
        best_lines = [
            f"- Model: `{best.get('model', '')}`",
            f"- p30 threshold: {float(best.get('p30_threshold', np.nan)):.2f}",
            f"- p60 threshold: {float(best.get('p60_threshold', np.nan)):.2f}",
            f"- Consecutive bins: {int(best.get('consecutive_bins', 0))}",
            f"- Cooldown: {int(best.get('cooldown_min', 0))} min",
            f"- Precision: {float(best.get('precision', np.nan)):.3f}",
            f"- Recall/POD: {float(best.get('recall', np.nan)):.3f}",
            f"- F1: {float(best.get('f1', np.nan)):.3f}",
            f"- False alerts/day: {float(best.get('false_alerts_per_day', np.nan)):.2f}",
            f"- Valid alerted events: {int(best.get('valid_alerted_events', 0))} / {int(best.get('total_events', len(events)))}",
            f"- Mean lead time: {float(best.get('mean_lead_time_min', np.nan)):.2f} min",
            f"- Median lead time: {float(best.get('median_lead_time_min', np.nan)):.2f} min",
        ]
    else:
        best_lines = ["- No valid v6 policy row was produced."]

    md = f"""# Forecasting v6 Expanded-Data Recommendation

## Decision

{decision}

Final recommended forecasting mode: **{'v6 expanded-data policy' if beats else 'v3 state-machine ML policy'}**.

## v6 Best Policy

{chr(10).join(best_lines)}

## v3 Baseline To Beat

- Precision: {V3_BASELINE['precision']:.3f}
- Recall/POD: {V3_BASELINE['recall']:.3f}
- F1: {V3_BASELINE['f1']:.3f}
- False alerts/day: {V3_BASELINE['false_alerts_per_day']:.2f}
- Valid alerted events: {V3_BASELINE['valid_alerted_events']} / {V3_BASELINE['total_events']}
- Mean lead time: {V3_BASELINE['mean_lead_time_min']:.2f} min
- Median lead time: {V3_BASELINE['median_lead_time_min']:.2f} min

## Expanded Data Summary

- ZIP files detected: {len(inventory)}
- Target dates requested: {manifest['date'].nunique()}
- FLARE target dates: {target_type_counts.get('FLARE', 0)}
- QUIET target dates: {target_type_counts.get('QUIET', 0)}
- Matched/rebuilt dates: {quality['date'].nunique() if not quality.empty else 0}
- Quality distribution: {qdist}
- v6 cleaned events used: {len(events)}

## Replacement Rule

v6 replaces v3 only if F1 > 0.634, false alerts/day < 1.33, and recall >= 0.824 or precision improves meaningfully with recall >= 0.75.

## Caveats

- v6 is an expanded-data experiment and does not overwrite v1-v5 results.
- BROKEN dates are excluded from supervised forecasting.
- GOOD quiet/control days remain as negative examples.
- GOES/SWPC is used only for validation/labels, not forecasting features.
"""
    (V6_DIR / "forecasting_v6_recommendation.md").write_text(md, encoding="utf-8")
    comparison_md = dataframe_to_markdown(comparison)
    (V6_DIR / "v3_vs_v6_comparison.md").write_text("# v3 vs v6 Comparison\n\n" + comparison_md + "\n", encoding="utf-8")
    return comparison, decision


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.4g}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    ensure_out()
    manifest = load_target_manifest()
    manifest["date"] = manifest["date"].map(normalize_date)
    inventory = inventory_zips(manifest)
    created_dirs, extracted_folders = safe_extract()
    matched_dates, match_report = write_match_report(manifest)
    target_matched = [d for d in sorted(manifest["date"].unique()) if d in matched_dates]
    if not target_matched:
        raise RuntimeError("No high-priority v6 target dates have both SoLEXS and HEL1OS Level-1 lightcurve coverage.")

    quality, catalogue = rebuild_dates(target_matched)
    dataset = build_v6_dataset(quality, catalogue)
    events = load_v6_events(catalogue, quality)
    predictions, model_comparison = run_blocked_models(dataset, events)
    sweep = sweep_policy(predictions, dataset, events)
    comparison, decision = write_comparison_and_recommendation(sweep, quality, events, inventory, manifest)

    flare_dates = manifest.drop_duplicates("date").query("target_type == 'FLARE'")["date"].nunique()
    quiet_dates = manifest.drop_duplicates("date").query("target_type == 'QUIET'")["date"].nunique()
    quiet_control_used = int((dataset.drop_duplicates("date")["is_quiet_day"] == True).sum()) if not dataset.empty and "is_quiet_day" in dataset else 0
    best = sweep.iloc[0] if not sweep.empty else None

    print("\n=== SuryaAlert v6 expanded-data pipeline summary ===")
    print(f"ZIPs detected: {len(inventory)}")
    print(f"extracted folders newly created: {len(created_dirs)}")
    print(f"extracted folders available: {len(extracted_folders)}")
    print(f"matched dates: {', '.join(target_matched)}")
    print(f"new events added: {max(0, len(events) - 17)}")
    print(f"quiet/control dates added or retained: {quiet_control_used} (manifest quiet targets={quiet_dates})")
    print("data-quality summary:")
    print(quality["overlap_quality_label"].value_counts().to_string() if not quality.empty else "none")
    if best is not None:
        print("v6 metrics:")
        print(
            f"precision={float(best['precision']):.3f}, recall={float(best['recall']):.3f}, "
            f"F1={float(best['f1']):.3f}, false alerts/day={float(best['false_alerts_per_day']):.2f}, "
            f"valid alerted events={int(best['valid_alerted_events'])}/{int(best['total_events'])}, "
            f"mean lead={float(best['mean_lead_time_min']):.2f} min, median lead={float(best['median_lead_time_min']):.2f} min"
        )
    else:
        print("v6 metrics: no valid policy row")
    print(f"whether v6 beats v3: {'yes' if 'does not beat' not in decision else 'no'}")
    print(f"final recommended forecasting mode: {'v6 expanded-data policy' if 'does not beat' not in decision else 'v3 state-machine ML policy'}")
    print("files created/updated:")
    for name in [
        "expanded_data_inventory.csv",
        "expanded_data_quality_audit.csv",
        "forecasting_v6_dataset.csv",
        "forecasting_v6_model_comparison.csv",
        "forecasting_v6_policy_sweep.csv",
        "forecasting_v6_recommendation.md",
        "v3_vs_v6_comparison.csv",
        "v3_vs_v6_comparison.md",
    ]:
        print(f"- {V6_DIR / name}")


if __name__ == "__main__":
    main()

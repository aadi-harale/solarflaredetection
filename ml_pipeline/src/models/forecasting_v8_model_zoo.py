from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object | None
    skipped_reason: str = ""


LABEL_PREFIXES = ("flare_next_", "time_to_")
BLOCKED_COLUMNS = {"timestamp", "source_date", "date", "quality_label", "is_quiet_day", "inside_detected_event"}


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    sampled = df.groupby("date").cumcount().mod(120).eq(0)
    hard = pd.to_numeric(df.get("hard_score", 0), errors="coerce").fillna(0).ge(4)
    fusion = pd.to_numeric(df.get("soft_hard_precursor_fusion_score", 0), errors="coerce").fillna(0).ge(4)
    return df[sampled | hard | fusion].copy().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in BLOCKED_COLUMNS or col.startswith(LABEL_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            cols.append(col)
    return cols


def model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            "logistic_regression_l2",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42),
            ),
        ),
        ModelSpec(
            "logistic_regression_balanced",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42),
            ),
        ),
        ModelSpec(
            "sgd_logistic_balanced",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0005, class_weight="balanced", max_iter=1000, tol=1e-3, random_state=42),
            ),
        ),
        ModelSpec(
            "extra_trees",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(n_estimators=70, max_depth=7, min_samples_leaf=18, class_weight="balanced", random_state=42, n_jobs=-1),
            ),
        ),
        ModelSpec(
            "random_forest_shallow",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(n_estimators=60, max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1),
            ),
        ),
        ModelSpec(
            "gradient_boosting",
            make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42),
            ),
        ),
        ModelSpec(
            "hist_gradient_boosting",
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=0.1, random_state=42),
            ),
        ),
    ]
    optional = [
        ("xgboost", "XGBoost skipped because optional package is not installed."),
        ("lightgbm", "LightGBM skipped because optional package is not installed."),
        ("catboost", "CatBoost skipped because optional package is not installed."),
    ]
    for package, reason in optional:
        if find_spec(package) is None:
            specs.append(ModelSpec(package, None, reason))
        else:
            specs.append(ModelSpec(package, None, f"{package} installed but not used in v8-Lite run to avoid adding external-model complexity."))
    return specs


def predict_scores(estimator: object, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_test)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(x_test)
        return 1.0 / (1.0 + np.exp(-raw))
    return estimator.predict(x_test)


def blocked_oof_predictions(df: pd.DataFrame, target: str, features: list[str], spec: ModelSpec) -> tuple[pd.DataFrame, str]:
    if spec.estimator is None:
        return pd.DataFrame(), spec.skipped_reason
    rows = []
    valid_folds = 0
    for date in sorted(df["date"].unique()):
        train = df[df["date"] != date]
        test = df[df["date"] == date]
        if test.empty or train[target].nunique() < 2:
            continue
        valid_folds += 1
        x_train = train[features].replace([np.inf, -np.inf], np.nan)
        x_test = test[features].replace([np.inf, -np.inf], np.nan)
        y_train = train[target].astype(int)
        spec.estimator.fit(x_train, y_train)
        score = predict_scores(spec.estimator, x_test)
        out = test[["timestamp", "date", target]].copy()
        out["model"] = spec.name
        out["target"] = target
        out["y_true"] = out[target].astype(int)
        out["score"] = score
        rows.append(out)
    if valid_folds < 2 or not rows:
        return pd.DataFrame(), "NOT_AVAILABLE_FAIRLY: fewer than two valid blocked folds."
    return pd.concat(rows, ignore_index=True), "AVAILABLE"


def rule_score_predictions(df: pd.DataFrame, target: str) -> pd.DataFrame:
    score = pd.to_numeric(df.get("hard_score", 0), errors="coerce").fillna(0) / 10.0
    out = df[["timestamp", "date", target]].copy()
    out["model"] = "rule_score_baseline"
    out["target"] = target
    out["y_true"] = out[target].astype(int)
    out["score"] = score.clip(0, 1)
    return out


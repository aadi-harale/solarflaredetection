from __future__ import annotations

import os


def is_dev_mode() -> bool:
    return os.environ.get("DEV_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def random_forest_estimators() -> int:
    return 50 if is_dev_mode() else 300

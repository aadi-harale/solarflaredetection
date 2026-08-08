from __future__ import annotations

import pandas as pd


JUNE03_START = pd.Timestamp("2026-06-03 00:00:00", tz="UTC")
JUNE03_END = pd.Timestamp("2026-06-04 00:00:00", tz="UTC")


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="mixed", errors="coerce")

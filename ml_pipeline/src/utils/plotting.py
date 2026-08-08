from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_lightcurves(df: pd.DataFrame, out_path: Path, date_label: str) -> None:
    solexs = df[df["instrument"] == "SoLEXS"]
    cdte = df[(df["instrument"] == "HEL1OS") & (df["detector"].astype(str).str.contains("CdTe", case=False))]
    czt = df[(df["instrument"] == "HEL1OS") & (df["detector"].astype(str).str.contains("CZT", case=False))]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for (det, band), group in solexs.groupby(["detector", "band"]):
        axes[0].plot(group["time_utc"], group["count_rate"], linewidth=0.8, label=f"{det} {band}")
    axes[0].set_title(f"SoLEXS Soft X-ray Light Curve - {date_label}")
    axes[0].set_ylabel("Counts/sec")
    axes[0].legend(fontsize=8)

    for (det, band), group in cdte.groupby(["detector", "band"]):
        if "1.8-90" in band:
            continue
        axes[1].plot(group["time_utc"], group["count_rate"], linewidth=0.7, label=f"{det} {band}")
    axes[1].set_title("HEL1OS CdTe Hard X-ray Light Curves")
    axes[1].set_ylabel("CTR")
    axes[1].legend(fontsize=7, ncol=2)

    for (det, band), group in czt.groupby(["detector", "band"]):
        if "18-160" in band:
            continue
        axes[2].plot(group["time_utc"], group["count_rate"], linewidth=0.7, label=f"{det} {band}")
    axes[2].set_title("HEL1OS CZT Hard X-ray Light Curves")
    axes[2].set_ylabel("CTR")
    axes[2].set_xlabel("UTC Time")
    axes[2].legend(fontsize=7, ncol=2)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot: {out_path}")


def plot_june03_lightcurves(df: pd.DataFrame, out_path: Path) -> None:
    plot_lightcurves(df, out_path, "2026-06-03")


def plot_nowcast_events(ts: pd.DataFrame, catalogue: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    axes[0].plot(ts.index, ts["soft_solexs_2_22"], linewidth=0.8, label="SoLEXS 2-22 keV")
    axes[0].set_title("SoLEXS Soft X-ray with Detected Events")
    axes[0].set_ylabel("Counts/sec")
    axes[0].legend()

    axes[1].plot(ts.index, ts["hard_cdte_5_20"], linewidth=0.7, label="HEL1OS CdTe 5-20 keV")
    axes[1].plot(ts.index, ts["hard_czt_20_40"], linewidth=0.7, label="HEL1OS CZT 20-40 keV")
    axes[1].set_title("HEL1OS Hard X-ray Bands")
    axes[1].set_ylabel("CTR")
    axes[1].legend()

    axes[2].plot(ts.index, ts["soft_score"], linewidth=0.7, label="Soft robust score")
    axes[2].plot(ts.index, ts["hard_score"], linewidth=0.7, label="Hard robust score")
    axes[2].axhline(8, linestyle="--", linewidth=1, label="Trigger threshold")
    axes[2].set_title("Robust Trigger Scores")
    axes[2].set_ylabel("Robust z-score")
    axes[2].set_xlabel("UTC Time")
    axes[2].legend()

    for _, row in catalogue.iterrows():
        for ax in axes:
            ax.axvspan(row["event_start"], row["event_end"], alpha=0.15)
            if pd.notna(row["hard_trigger_time"]):
                ax.axvline(row["hard_trigger_time"], linestyle=":", linewidth=1)
            ax.axvline(row["soft_peak_time"], linestyle="--", linewidth=1)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[SAVED] {out_path}")

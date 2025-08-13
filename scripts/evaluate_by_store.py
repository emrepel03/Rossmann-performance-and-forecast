# scripts/evaluate_by_store.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def group_metrics(df: pd.DataFrame, by_cols):
    # overall metrics without groupby
    if not by_cols:
        return pd.DataFrame([{
            "n": len(df),
            "rmspe": rmspe(df["y_true"].values, df["y_pred"].values),
            "rmse": rmse(df["y_true"].values, df["y_pred"].values),
            "mape": mape(df["y_true"].values, df["y_pred"].values),
            "mean_true": float(np.mean(df["y_true"].values)),
        }])

    # grouped metrics (future-proof: exclude grouping columns from apply)
    out = (
        df.groupby(by_cols, dropna=False, observed=True)
          .apply(lambda g: pd.Series({
              "n": len(g),
              "rmspe": rmspe(g["y_true"].values, g["y_pred"].values),
              "rmse": rmse(g["y_true"].values, g["y_pred"].values),
              "mape": mape(g["y_true"].values, g["y_pred"].values),
              "mean_true": float(np.mean(g["y_true"].values)),
          }), include_groups=False)
          .reset_index()
    )
    return out


def main(
    preds_path: Path,
    wide_path: Path,
    out_metrics_dir: Path,
    out_fig_dir: Path,
):
    ensure_dirs(out_metrics_dir, out_fig_dir)

    # Load predictions (has Store/Date if present in X_test at training time)
    preds = pd.read_csv(preds_path, low_memory=False)
    if "Date" in preds.columns:
        preds["Date"] = pd.to_datetime(preds["Date"])

    # Bring in extra context (Promo, etc.) from the engineered wide table
    wide = pd.read_csv(wide_path, low_memory=False)
    if "Date" in wide.columns:
        wide["Date"] = pd.to_datetime(wide["Date"])

    merge_keys = [c for c in ["Store", "Date"] if c in preds.columns and c in wide.columns]
    if merge_keys:
        preds = preds.merge(
            wide[merge_keys + [c for c in ["Promo", "DayOfWeek"] if c in wide.columns]],
            on=merge_keys, how="left",
        )

    # ---------- Metrics ----------
    # by store
    by_store = group_metrics(preds, ["Store"] if "Store" in preds.columns else [])
    by_store = by_store.sort_values("rmspe", ascending=False)
    by_store.to_csv(out_metrics_dir / "by_store.csv", index=False)

    # by promo (if available)
    if "Promo" in preds.columns:
        by_promo = group_metrics(preds, ["Promo"])
        by_promo.to_csv(out_metrics_dir / "by_promo.csv", index=False)

    # by weekday
    if "Date" in preds.columns:
        preds["Weekday"] = preds["Date"].dt.day_name()
        by_weekday = group_metrics(preds, ["Weekday"])
        # Put weekdays in logical order
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        by_weekday["Weekday"] = pd.Categorical(by_weekday["Weekday"], order, ordered=True)
        by_weekday = by_weekday.sort_values("Weekday")
        by_weekday.to_csv(out_metrics_dir / "by_weekday.csv", index=False)

    # ---------- Plots ----------
    # (1) Daily aggregate actual vs predicted (sum over all stores)
    if "Date" in preds.columns:
        daily = preds.groupby("Date", as_index=False)[["y_true", "y_pred"]].sum()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(daily["Date"], daily["y_true"], label="Actual (sum)")
        ax.plot(daily["Date"], daily["y_pred"], label="Predicted (sum)")
        ax.set_title("Daily Aggregate – Actual vs Predicted")
        ax.set_xlabel("Date"); ax.set_ylabel("Sales (sum)")
        ax.legend(); fig.tight_layout()
        fig.savefig(out_fig_dir / "agg_daily_actual_vs_pred.png", dpi=140)
        plt.close(fig)

    # (2) Top-20 worst stores by RMSPE (barh)
    if "Store" in by_store.columns and len(by_store) > 0:
        top = by_store.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(top["Store"].astype(str), top["rmspe"])
        ax.set_title("Worst 20 Stores by RMSPE (lower is better)")
        ax.set_xlabel("RMSPE")
        fig.tight_layout()
        fig.savefig(out_fig_dir / "store_rmspe_top20.png", dpi=140)
        plt.close(fig)

    # Print a tiny summary
    overall = group_metrics(preds, [])[["rmspe", "rmse", "mape"]].iloc[0].to_dict()
    print("\nOverall (test window):", overall)
    print(f"Saved:\n- {out_metrics_dir/'by_store.csv'}")
    if (out_metrics_dir / "by_promo.csv").exists(): print(f"- {out_metrics_dir/'by_promo.csv'}")
    if (out_metrics_dir / "by_weekday.csv").exists(): print(f"- {out_metrics_dir/'by_weekday.csv'}")
    print(f"- {out_fig_dir/'agg_daily_actual_vs_pred.png'}")
    if (out_fig_dir / "store_rmspe_top20.png").exists(): print(f"- {out_fig_dir/'store_rmspe_top20.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds_path", type=str, default="reports/predictions/lgbm_test_predictions.csv")
    p.add_argument("--wide_path", type=str, default="data/processed/rossmann_features_wide.csv")
    p.add_argument("--out_metrics_dir", type=str, default="reports/metrics")
    p.add_argument("--out_fig_dir", type=str, default="reports/figures")
    args = p.parse_args()

    main(
        preds_path=Path(args.preds_path),
        wide_path=Path(args.wide_path),
        out_metrics_dir=Path(args.out_metrics_dir),
        out_fig_dir=Path(args.out_fig_dir),
    )
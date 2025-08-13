# scripts/compare_models.py
# Compare LightGBM (existing preds) vs Prophet on the same test window.
# Saves:
#   reports/metrics/model_compare_overall.csv
#   reports/metrics/model_compare_by_store.csv
#   reports/figures/model_compare_overall_rmspe_bar.png
#   reports/figures/model_compare_top20_rmspe_bars.png
#   reports/figures/model_compare_lines_store_<id>.png

from __future__ import annotations
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prophet lazy import (already in your venv)
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# -----------------------
# Metrics
# -----------------------
def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

# -----------------------
# Prophet per store on test window
# -----------------------
def prophet_forecast_for_test(df_store: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.DataFrame:
    """df_store has at least Date, Sales, Promo (optional). Returns DataFrame with Date,yhat for the test window."""
    # Prepare Prophet columns
    tmp = df_store.rename(columns={"Date": "ds", "Sales": "y"}).copy()
    has_promo = "Promo" in tmp.columns

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.5,   # a little flexible but fast
    )
    if has_promo:
        m.add_regressor("Promo")

    fit_df = tmp[tmp["ds"] < test_start]  # train before test window
    # If a store has too few points, bail out
    if len(fit_df) < 30:
        return pd.DataFrame(columns=["Date", "yhat"])

    cols = ["ds", "y"] + (["Promo"] if has_promo else [])
    m.fit(fit_df[cols])

    # Forecast exactly on test dates
    future = tmp[(tmp["ds"] >= test_start) & (tmp["ds"] <= test_end)][["ds"]].copy()
    if has_promo:
        future["Promo"] = tmp.set_index("ds").loc[future["ds"], "Promo"].values

    fcst = m.predict(future)
    out = fcst[["ds", "yhat"]].rename(columns={"ds": "Date"})
    return out

# -----------------------
# Main
# -----------------------
def main(wide_path: Path, preds_path: Path, out_dir: Path, n_lines: int = 5):
    out_dir_fig = Path("reports/figures")
    out_dir_metrics = Path("reports/metrics")
    out_dir_fig.mkdir(parents=True, exist_ok=True)
    out_dir_metrics.mkdir(parents=True, exist_ok=True)

    # Load LightGBM predictions (your file layout)
    lgbm = pd.read_csv(preds_path, parse_dates=["Date"])
    lgbm.rename(columns={"y_true": "Sales", "y_pred": "Pred_LGBM"}, inplace=True)

    # Determine test window from the predictions file
    test_start, test_end = lgbm["Date"].min(), lgbm["Date"].max()

    # Load wide data and keep only columns we need
    df = pd.read_csv(wide_path, low_memory=False, parse_dates=["Date"])
    need_cols = ["Store", "Date", "Sales", "Promo"]
    have_cols = [c for c in need_cols if c in df.columns]
    df = df[have_cols].copy()

    # Limit to the stores & dates present in test predictions
    stores = lgbm["Store"].unique()
    df = df[df["Store"].isin(stores)]

    # Compute LGBM metrics by store
    by_store_lgbm = (
        lgbm.groupby("Store")
            .apply(lambda g: pd.Series({
                "rmspe_lgbm": rmspe(g["Sales"], g["Pred_LGBM"]),
                "rmse_lgbm": rmse(g["Sales"], g["Pred_LGBM"]),
                "mape_lgbm": mape(g["Sales"], g["Pred_LGBM"]),
                "sales_sum_test": g["Sales"].sum()
            }))
            .reset_index()
    )

    # Prophet per store on the same window
    if not HAS_PROPHET:
        print("⚠️ prophet not installed; skipping Prophet comparison.")
        by_store = by_store_lgbm.copy()
        by_store["rmspe_prophet"] = np.nan
        by_store["rmse_prophet"] = np.nan
        by_store["mape_prophet"] = np.nan
    else:
        rows = []
        for s in stores:
            hist = df[df["Store"] == s].sort_values("Date")
            # Need training history prior to test_start
            if (hist["Date"] < test_start).sum() < 30:
                # Not enough history; skip Prophet
                yhat = pd.DataFrame({"Date": lgbm[lgbm["Store"] == s]["Date"], "yhat": np.nan})
            else:
                yhat = prophet_forecast_for_test(hist, test_start, test_end)

            # Merge with actuals for test window
            test_truth = hist[(hist["Date"] >= test_start) & (hist["Date"] <= test_end)][["Date", "Sales"]]
            merged = test_truth.merge(yhat, on="Date", how="left")
            if merged["yhat"].notna().sum() == 0:
                ms = {"rmspe_prophet": np.nan, "rmse_prophet": np.nan, "mape_prophet": np.nan}
            else:
                ms = {
                    "rmspe_prophet": rmspe(merged["Sales"].values, merged["yhat"].values),
                    "rmse_prophet": rmse(merged["Sales"].values, merged["yhat"].values),
                    "mape_prophet": mape(merged["Sales"].values, merged["yhat"].values),
                }
            ms["Store"] = s
            ms["sales_sum_test"] = test_truth["Sales"].sum()
            rows.append(ms)
        by_store_prophet = pd.DataFrame(rows)

        # Join LGBM + Prophet
        by_store = by_store_lgbm.merge(by_store_prophet, on=["Store", "sales_sum_test"], how="left")

    # Overall metrics (sales-weighted over test period)
    def _overall_weighted(dfm, pred_col):
        # weight by actual sales to approximate portfolio impact
        w = dfm.groupby("Store")["sales_sum_test"].first()
        m = dfm.set_index("Store")
        return {
            "rmspe": np.average(m[pred_col], weights=w) if w.notna().all() else m[pred_col].mean(),
        }

    overall = {
        "lightgbm_rmspe": rmspe(lgbm["Sales"], lgbm["Pred_LGBM"]),
        "prophet_rmspe": np.nan if by_store["rmspe_prophet"].isna().all()
            else np.average(by_store["rmspe_prophet"], weights=by_store["sales_sum_test"]),
        "lightgbm_rmse": rmse(lgbm["Sales"], lgbm["Pred_LGBM"]),
        "lightgbm_mape": mape(lgbm["Sales"], lgbm["Pred_LGBM"]),
    }

    # Save CSVs
    by_store.sort_values("rmspe_lgbm", inplace=True)
    by_store.to_csv(out_dir_metrics / "model_compare_by_store.csv", index=False)
    pd.DataFrame([{
        "test_start": test_start.date(),
        "test_end": test_end.date(),
        **overall
    }]).to_csv(out_dir_metrics / "model_compare_overall.csv", index=False)

    # ---------- Plots ----------
    # 1) Overall RMSPE bar (LGBM vs Prophet)
    fig, ax = plt.subplots(figsize=(5,4))
    methods = ["LightGBM", "Prophet"]
    vals = [
        overall["lightgbm_rmspe"],
        overall["prophet_rmspe"]
    ]
    ax.bar(methods, vals)
    ax.set_ylabel("RMSPE (lower is better)")
    ax.set_title("Overall RMSPE – LightGBM vs Prophet")
    for i,v in enumerate(vals):
        if pd.notna(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        else:
            ax.text(i, 0.02, "n/a", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir_fig / "model_compare_overall_rmspe_bar.png", dpi=140)
    plt.close(fig)

    # 2) Top-20 stores, grouped bars (RMSPE)
    top20 = by_store.copy()
    # choose by total sales in test to emphasize important stores
    top20 = top20.sort_values("sales_sum_test", ascending=False).head(20)
    x = np.arange(len(top20))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(x - width/2, top20["rmspe_lgbm"], width, label="LightGBM")
    ax.bar(x + width/2, top20["rmspe_prophet"], width, label="Prophet")
    ax.set_xticks(x)
    ax.set_xticklabels(top20["Store"].astype(str), rotation=0)
    ax.set_xlabel("Store")
    ax.set_ylabel("RMSPE")
    ax.set_title("RMSPE by Store (Top-20 by Sales) – LightGBM vs Prophet")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir_fig / "model_compare_top20_rmspe_bars.png", dpi=140)
    plt.close(fig)

    # 3) Representative lines (best / median / worst by LGBM RMSPE)
    reps = []
    if len(by_store) >= 3:
        reps = [
            by_store.iloc[0]["Store"],                                      # best
            by_store.iloc[len(by_store)//2]["Store"],                       # median-ish
            by_store.iloc[-1]["Store"],                                     # worst
        ]
    else:
        reps = by_store["Store"].tolist()

    for s in reps[:n_lines]:
        # assemble an all-in-one frame with actual, lgbm, prophet
        truth = df[(df["Store"] == s) & (df["Date"] >= test_start) & (df["Date"] <= test_end)][["Date", "Sales"]]
        lgbm_s = lgbm[lgbm["Store"] == s][["Date", "Pred_LGBM"]]
        merged = truth.merge(lgbm_s, on="Date", how="left").sort_values("Date")

        if HAS_PROPHET:
            # recompute Prophet yhat for this store quickly (could also cache above if needed)
            hist = df[df["Store"] == s].sort_values("Date")
            yhat = prophet_forecast_for_test(hist, test_start, test_end)
            merged = merged.merge(yhat.rename(columns={"yhat": "Pred_Prophet"}), on="Date", how="left")

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(merged["Date"], merged["Sales"], label="Actual")
        ax.plot(merged["Date"], merged["Pred_LGBM"], label="LightGBM")
        if "Pred_Prophet" in merged.columns:
            ax.plot(merged["Date"], merged["Pred_Prophet"], label="Prophet")
        ax.set_title(f"Store {s} – Actual vs Forecast (Test Window)")
        ax.set_xlabel("Date"); ax.set_ylabel("Sales")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir_fig / f"model_compare_lines_store_{s}.png", dpi=140)
        plt.close(fig)

    print("Saved:")
    print(f"- {out_dir_metrics/'model_compare_overall.csv'}")
    print(f"- {out_dir_metrics/'model_compare_by_store.csv'}")
    print(f"- {out_dir_fig/'model_compare_overall_rmspe_bar.png'}")
    print(f"- {out_dir_fig/'model_compare_top20_rmspe_bars.png'}")
    for s in reps[:n_lines]:
        print(f"- {out_dir_fig/f'model_compare_lines_store_{s}.png'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wide_path", type=str, default="data/processed/rossmann_features_wide.csv")
    p.add_argument("--preds_path", type=str, default="reports/predictions/lgbm_test_predictions.csv")
    p.add_argument("--out_dir", type=str, default="reports/figures")
    p.add_argument("--n_lines", type=int, default=5)
    args = p.parse_args()
    main(Path(args.wide_path), Path(args.preds_path), Path(args.out_dir), n_lines=args.n_lines)
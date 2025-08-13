# scripts/promo_simulator.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean(((y_true[mask]-y_pred[mask])/y_true[mask])**2)))

def align_to_model_features(df: pd.DataFrame, model_feat_names: list[str]) -> pd.DataFrame:
    X = df.copy()
    # Add missing features as 0
    for f in model_feat_names:
        if f not in X.columns:
            X[f] = 0
    # Keep only model features (preserve order)
    X = X[model_feat_names]
    # LightGBM expects numeric
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X

def load_model(model_path: Path):
    model = load(model_path)
    # Works for LightGBM Booster and sklearn wrappers
    try:
        feat_names = model.feature_name()
    except Exception:
        feat_names = getattr(model, "feature_name_", None) or getattr(model, "feature_names_", None)
    if feat_names is None:
        raise RuntimeError("Could not read feature names from the saved model.")
    return model, list(feat_names)

def recompute_simple_interactions(df: pd.DataFrame):
    """Recompute a couple of simple Promo interactions if they exist."""
    if "Promo" in df.columns:
        if "Month" in df.columns and "Promo_Month" in df.columns:
            df["Promo_Month"] = df["Promo"] * df["Month"]
        # Common naming variants for a numeric store-type code
        base_codes = [c for c in ["StoreType_code", "StoreTypeCode", "StoreTypeNum"] if c in df.columns]
        if base_codes and "Promo_StoreType" in df.columns:
            df["Promo_StoreType"] = df["Promo"] * df[base_codes[0]]
    return df

def main(
    wide_path: Path,
    model_path: Path,
    out_dir: Path,
    start: str,
    end: str,
    store_ids: str | None,
    stores_top_k: int | None
):
    ensure_dirs(out_dir)

    df = pd.read_csv(wide_path, low_memory=False)
    if "Date" not in df.columns:
        raise ValueError("Expected 'Date' in wide CSV.")
    df["Date"] = pd.to_datetime(df["Date"])

    # Subset window
    mask = (df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))
    dfw = df.loc[mask].copy()
    if dfw.empty:
        raise ValueError("No rows in the requested date window.")

    # Limit scope: either explicit store list or top-K by sales in window
    if store_ids:
        wanted = set(int(s) for s in store_ids.split(","))
        dfw = dfw[dfw["Store"].isin(wanted)]
    elif stores_top_k:
        top = (
            dfw.groupby("Store")["Sales"].sum().sort_values(ascending=False).head(stores_top_k).index
            if "Sales" in dfw.columns else
            dfw["Store"].value_counts().head(stores_top_k).index
        )
        dfw = dfw[dfw["Store"].isin(top)]

    # Only consider open days
    if "Open" in dfw.columns:
        dfw = dfw[dfw["Open"] == 1].copy()

    # Load model and features
    model, model_feats = load_model(model_path)

    # Baseline predictions (as-is)
    X_base = align_to_model_features(dfw, model_feats)
    y_base = model.predict(X_base)

    # Counterfactual: set Promo=1 where it was 0; keep 1's as 1 to avoid double counting
    if "Promo" not in dfw.columns:
        raise ValueError("Column 'Promo' not found; required for simulation.")
    dfx = dfw.copy()
    dfx["Promo_cf"] = 1
    # only flip where originally 0
    dfx["Promo_cf"] = np.where(dfx["Promo"] == 0, 1, dfx["Promo"])
    dfx["Promo"] = dfx["Promo_cf"]  # overwrite for counterfactual

    # Recompute simple interactions that depend on Promo (if present)
    dfx = recompute_simple_interactions(dfx)

    X_cf = align_to_model_features(dfx, model_feats)
    y_cf = model.predict(X_cf)

    out = dfw[["Store", "Date"]].copy()
    out["pred_baseline"] = y_base
    out["pred_with_promo"] = y_cf
    out["lift_abs"] = out["pred_with_promo"] - out["pred_baseline"]
    out["lift_pct"] = np.where(out["pred_baseline"] != 0, out["lift_abs"] / out["pred_baseline"], 0.0)

    # Save detailed lifts
    detail_path = out_dir / "lift_by_store_day.csv"
    out.to_csv(detail_path, index=False)

    # Summaries
    by_store = (
        out.groupby("Store")
           .agg(days=("Date", "nunique"),
                lift_total=("lift_abs", "sum"),
                lift_mean=("lift_abs", "mean"),
                lift_pct_mean=("lift_pct", "mean"))
           .reset_index()
           .sort_values("lift_total", ascending=False)
    )
    by_store_path = out_dir / "lift_summary_by_store.csv"
    by_store.to_csv(by_store_path, index=False)

    # Plot top-12 stores by total lift
    fig_dir = out_dir
    ensure_dirs(fig_dir)
    top12 = by_store.head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top12["Store"].astype(str), top12["lift_total"])
    ax.set_title("Promo What-If: Total Lift by Store (Top 12)")
    ax.set_xlabel("Store")
    ax.set_ylabel("Total Lift (sales units)")
    fig.tight_layout()
    fig.savefig(fig_dir / "lift_top12_stores.png", dpi=140)
    plt.close(fig)

    # Optional sanity if actual Sales present in window
    overall = {}
    if "Sales" in dfw.columns:
        overall = {
            "window_start": pd.to_datetime(start).date(),
            "window_end": pd.to_datetime(end).date(),
            "rmspe_baseline": rmspe(dfw["Sales"].values, y_base),
        }
        pd.DataFrame([overall]).to_csv(out_dir / "sanity_overall.csv", index=False)

    print("Saved:")
    print(f"- {detail_path}")
    print(f"- {by_store_path}")
    print(f"- {fig_dir / 'lift_top12_stores.png'}")
    if overall:
        print(f"- {out_dir / 'sanity_overall.csv'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wide_path", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="reports/promo_sim")
    p.add_argument("--start", type=str, required=True)  # e.g. 2015-06-20
    p.add_argument("--end", type=str, required=True)    # e.g. 2015-07-31
    p.add_argument("--store_ids", type=str, default=None, help="Comma-separated store ids to include")
    p.add_argument("--stores_top_k", type=int, default=25)
    args = p.parse_args()

    main(
        wide_path=Path(args.wide_path),
        model_path=Path(args.model_path),
        out_dir=Path(args.out_dir),
        start=args.start,
        end=args.end,
        store_ids=args.store_ids,
        stores_top_k=args.stores_top_k,
    )
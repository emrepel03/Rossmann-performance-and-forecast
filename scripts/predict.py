# scripts/predict.py
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def main(model_path: Path, input_csv: Path, output_csv: Path):
    model = joblib.load(model_path)

    # Load features
    df = pd.read_csv(input_csv, low_memory=False)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # Use the trained model's feature names as the contract
    feature_names = list(model.feature_name())
    missing = [c for c in feature_names if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names]

    if missing:
        # Create safe zeros for unseen one-hots / engineered cols
        for c in missing:
            df[c] = 0.0

    # Keep order identical to training
    X = df[feature_names].copy()

    # Coerce dtypes to float (LightGBM is fine with float)
    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(np.int8)
    X = X.astype(float)

    best_iter = getattr(model, "best_iteration", None)
    preds = model.predict(X, num_iteration=best_iter)

    # Build output with useful identifiers if available
    id_cols = [c for c in ["Store", "Date"] if c in df.columns]
    out = df[id_cols].copy() if id_cols else pd.DataFrame({"row": np.arange(len(preds))})
    out["prediction"] = preds

    ensure_dirs(Path(output_csv).parent)
    out.to_csv(output_csv, index=False)

    # Quick stdout summary
    print(f"Read {len(df):,} rows from {input_csv}")
    print(f"Missing features filled with 0: {len(missing)} -> {missing[:10]}{'...' if len(missing)>10 else ''}")
    if extra:
        print(f"Ignored extra columns (not in model): {len(extra)} -> {extra[:10]}{'...' if len(extra)>10 else ''}")
    print(f"Saved predictions -> {output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/lightgbm_sales.pkl")
    p.add_argument("--input_csv", type=str, required=True,
                   help="CSV with the same engineered features as training (can include extra columns).")
    p.add_argument("--output_csv", type=str, default="reports/predictions/new_predictions.csv")
    args = p.parse_args()

    main(
        model_path=Path(args.model_path),
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
    )
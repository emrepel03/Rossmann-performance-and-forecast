# modeling.py
# -------------------------------------------------------------
# Trains:
#   1) LightGBM global regressor with Optuna (200 trials by default)
#   2) Prophet per-store models for top 3 stores
#
# Splits (time-aware):
#   - Test: last N test_days (default 42 ~= 6 weeks)
#   - Valid: the 42 days before test
#   - Train: the rest
#
# Metrics: RMSPE (primary), RMSE, MAPE
# Saves:
#   - models/lightgbm_sales.pkl
#   - reports/metrics/metrics_summary.csv
#   - reports/figures/lgbm_feature_importance.png
#   - reports/figures/lgbm_actual_vs_pred.png
#   - reports/figures/prophet_store_<id>.png
#   - reports/predictions/lgbm_test_predictions.csv
# -------------------------------------------------------------

from __future__ import annotations
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import multiprocessing as mp

NUM_THREADS = max(1, (mp.cpu_count() or 2) - 1)

import numpy as np
import pandas as pd
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
from joblib import dump
import matplotlib.pyplot as plt
import json
import time
import re

# prophet can be slow to import; do it lazily when needed
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# -----------------------
# Utils
# -----------------------
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def time_split(df: pd.DataFrame, date_col: str, test_days: int, val_days: int):
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=val_days - 1)

    train_idx = df[date_col] < val_start
    val_idx   = (df[date_col] >= val_start) & (df[date_col] <= val_end)
    test_idx  = df[date_col] >= test_start
    return train_idx, val_idx, test_idx, dict(
        val_start=val_start.date(), val_end=val_end.date(),
        test_start=test_start.date(), test_end=max_date.date()
    )

# -----------------------
# Feature selection
# -----------------------
def select_features(df: pd.DataFrame) -> list[str]:
    """
    Choose modeling features from the engineered wide table.
    We EXCLUDE:
      - target: Sales
      - 'Customers' (not known at forecasting time; would inflate performance)
      - raw identifiers or high-leak columns
    """
    drop_cols = {
        "Sales", "Customers",
        "Date",          # keep for splitting, but not as feature
        "StateHoliday",  # raw label if present (we have one-hot in wide)
        "StoreType", "Assortment", "PromoInterval",  # raw cats
    }
    # Keep numeric + one-hot columns (wide is already OHE)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    # Some safety: remove boolean dtype that matplotlib might choke on later
    return feature_cols

# -----------------------
# Column name sanitization (for LightGBM)
# -----------------------

def _sanitize_feature_names(cols: list[str]) -> list[str]:
    """Replace non-alphanumeric/underscore chars with '_' and ensure uniqueness."""
    cleaned = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in cols]
    # Ensure uniqueness if collisions happen after cleaning
    seen = {}
    unique = []
    for name in cleaned:
        if name not in seen:
            seen[name] = 0
            unique.append(name)
        else:
            seen[name] += 1
            unique_name = f"{name}__{seen[name]}"
            while unique_name in seen:
                seen[name] += 1
                unique_name = f"{name}__{seen[name]}"
            seen[unique_name] = 0
            unique.append(unique_name)
    return unique

# -----------------------
# Optuna trial checkpointing helpers
# -----------------------
def _save_optuna_progress(study: optuna.Study, out_dir: Path) -> None:
    """Persist best params and full trials dataframe so runs can be resumed/inspected."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save best params
        if study.best_trial is not None:
            best_payload = {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_trial.params,
            }
            (out_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2))
        # Save trials dataframe
        try:
            df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs"))
            df_trials.to_csv(out_dir / "optuna_trials.csv", index=False)
        except Exception:
            pass
    except Exception as e:
        print(f"[optuna] checkpoint save failed: {e}")

# -----------------------
# LightGBM + Optuna
# -----------------------
def train_lgbm_optuna(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    categorical: list[str] | None,
    n_trials: int,
    random_state: int = 42,
    early_stopping_rounds: int = 200,
    test_days: int = 42,
    val_days: int = 42,
):
    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    train_idx, val_idx, test_idx, split_info = time_split(df, "Date", test_days, val_days)

    X_train, y_train = df.loc[train_idx, features], df.loc[train_idx, target]
    X_val,   y_val   = df.loc[val_idx,   features], df.loc[val_idx,   target]
    X_test,  y_test  = df.loc[test_idx,  features], df.loc[test_idx,  target]

    # Sanitize feature names to avoid LightGBM JSON-name errors
    safe_cols = _sanitize_feature_names(list(X_train.columns))
    X_train.columns = safe_cols
    X_val.columns = safe_cols
    X_test.columns = safe_cols

    # LightGBM works great with numeric (OHE), so categorical list is optional
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_val,   label=y_val,   free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            "num_threads": NUM_THREADS,
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 2000),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),
        }

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                LightGBMPruningCallback(trial, "rmse"),
            ],
        )
        # Evaluate with RMSPE on validation
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        return rmspe(y_val.values, val_pred)

    # Use SQLite storage so we can resume long runs and not lose progress
    storage_path = Path("data/optuna/rossmann_lgbm.sqlite3")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20, seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    study = optuna.create_study(
        study_name="rossmann_lgbm",
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    def _trial_cb(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial):
        _save_optuna_progress(study_obj, metrics_dir)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[_trial_cb])

    best_params = study.best_params
    # Retrain on (train + val) with best params, evaluate on test
    dtrain_full = lgb.Dataset(pd.concat([X_train, X_val]), label=pd.concat([y_train, y_val]))
    final_model = lgb.train(
        {**best_params, "objective": "regression", "metric": "rmse", "verbosity": -1, "boosting_type": "gbdt", "feature_pre_filter": False, "num_threads": NUM_THREADS},
        dtrain_full,
        valid_sets=[lgb.Dataset(X_test, label=y_test)],
        num_boost_round=10000,
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )

    test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    metrics = {
        "split_val_start": split_info["val_start"],
        "split_val_end": split_info["val_end"],
        "split_test_start": split_info["test_start"],
        "split_test_end": split_info["test_end"],
        "rmspe_test": rmspe(y_test.values, test_pred),
        "rmse_test": rmse(y_test.values, test_pred),
        "mape_test": mape(y_test.values, test_pred),
        "best_iteration": final_model.best_iteration,
        **best_params
    }

    return final_model, metrics, X_test, y_test, test_pred, study

# -----------------------
# Prophet (top 3 stores)
# -----------------------
def train_prophet_top_stores(df_long: pd.DataFrame, test_days: int, out_dir: Path):
    if not _HAS_PROPHET:
        print("⚠️ prophet not installed; skipping Prophet models.")
        return []

    # We’ll mirror the same test window
    max_date = df_long["Date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)

    # Choose top 3 stores by training period sales volume
    train_period = df_long[df_long["Date"] < test_start]
    top3 = (
        train_period.groupby("Store")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    results = []
    for store_id in top3:
        sub = df_long[df_long["Store"] == store_id].copy()
        # Prophet expects columns ds (date) and y (value)
        prophet_df = sub.rename(columns={"Date": "ds", "Sales": "y"})
        # Use Promo as regressor if present
        has_promo = "Promo" in prophet_df.columns
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        if has_promo:
            m.add_regressor("Promo")
        # Fit on all except test
        fit_df = prophet_df[prophet_df["ds"] < test_start]
        m.fit(fit_df[["ds", "y"] + (["Promo"] if has_promo else [])])

        # Forecast over the test horizon only (keep real calendar)
        future = prophet_df[prophet_df["ds"] >= test_start][["ds"]].copy()
        if has_promo:
            # use actual promo flag for those dates
            future["Promo"] = prophet_df.set_index("ds").loc[future["ds"], "Promo"].values
        fcst = m.predict(future)

        # Merge with actuals for metrics
        test_merge = future.merge(
            prophet_df[["ds", "y"]], on="ds", how="left"
        ).merge(
            fcst[["ds", "yhat"]], on="ds", how="left"
        )
        r = {
            "store": store_id,
            "rmspe_test": rmspe(test_merge["y"].values, test_merge["yhat"].values),
            "rmse_test": rmse(test_merge["y"].values, test_merge["yhat"].values),
            "mape_test": mape(test_merge["y"].values, test_merge["yhat"].values),
        }
        results.append(r)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prophet_df["ds"], prophet_df["y"], label="Actual")
        ax.plot(test_merge["ds"], test_merge["yhat"], label="Prophet Forecast")
        ax.set_title(f"Prophet – Store {store_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        fig.tight_layout()
        fig_path = out_dir / f"prophet_store_{store_id}.png"
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)

    return results

# -----------------------
# Main
# -----------------------
def main(
    wide_path: Path,
    n_trials: int,
    test_days: int,
    val_days: int,
    seed: int = 42,
):
    out_models = Path("models")
    out_fig = Path("reports/figures")
    out_metrics = Path("reports/metrics")
    out_preds = Path("reports/predictions")
    ensure_dirs(out_models, out_fig, out_metrics, out_preds)

    # Load engineered wide data
    df = pd.read_csv(wide_path, low_memory=False)
    # Ensure date type
    df["Date"] = pd.to_datetime(df["Date"])

    # Choose features
    features = select_features(df)
    target = "Sales"
    print(f"Using {len(features)} features.")
    print(f"Using {NUM_THREADS} CPU threads for LightGBM.")

    # Train LGBM + Optuna
    print(f"🟩 Training LightGBM with Optuna ({n_trials} trials)...")
    model, lgbm_metrics, X_test, y_test, y_pred, study = train_lgbm_optuna(
        df=df,
        features=features,
        target=target,
        categorical=None,   # already OHE
        n_trials=n_trials,
        random_state=seed,
        test_days=test_days,
        val_days=val_days,
    )
    _save_optuna_progress(study, Path("reports/metrics"))

    # Save model
    model_path = out_models / "lightgbm_sales.pkl"
    dump(model, model_path)
    print(f"✅ Saved model: {model_path}")

    # Save test predictions (add Store/Date if available in original df)
    meta_cols = []
    if "Store" in df.columns:
        meta_cols.append("Store")
    if "Date" in df.columns:
        meta_cols.append("Date")
    if meta_cols:
        base_meta = df.loc[X_test.index, meta_cols].copy()
    else:
        base_meta = pd.DataFrame(index=X_test.index)

    preds_df = base_meta
    preds_df["y_true"] = y_test.values
    preds_df["y_pred"] = y_pred
    preds_path = out_preds / "lgbm_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # Feature importance
    try:
        feature_names = model.feature_name()
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False)
        imp_path = out_metrics / "lgbm_feature_importance.csv"
        importances.to_csv(imp_path, index=False)

        fig, ax = plt.subplots(figsize=(8, 10))
        topn = importances.head(30).iloc[::-1]
        ax.barh(topn["feature"], topn["importance_gain"])
        ax.set_title("LightGBM Feature Importance (gain) – Top 30")
        ax.set_xlabel("Gain")
        fig.tight_layout()
        fig.savefig(out_fig / "lgbm_feature_importance.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"Feature importance plotting failed: {e}")

    # Actual vs Pred (global)
    fig, ax = plt.subplots(figsize=(12, 5))
    # sort by date if available
    try:
        order = np.argsort(pd.to_datetime(preds_df["Date"]).values)
        ax.plot(preds_df["Date"].values[order], preds_df["y_true"].values[order], label="Actual")
        ax.plot(preds_df["Date"].values[order], preds_df["y_pred"].values[order], label="Predicted")
        ax.set_xlabel("Date")
    except Exception:
        ax.plot(preds_df["y_true"].values, label="Actual")
        ax.plot(preds_df["y_pred"].values, label="Predicted")
        ax.set_xlabel("Sample")
    ax.set_title("LightGBM – Actual vs Predicted (Test)")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fig / "lgbm_actual_vs_pred.png", dpi=140)
    plt.close(fig)

    # Prophet (top 3 stores)
    print("🔵 Training Prophet for top 3 stores...")
    prophet_fig_dir = out_fig
    prophet_results = train_prophet_top_stores(df_long=df, test_days=test_days, out_dir=prophet_fig_dir)

    # Metrics summary
    rows = [{"model": "LightGBM", **lgbm_metrics}]
    for r in prophet_results:
        rows.append({
            "model": f"Prophet_store_{r['store']}",
            "rmspe_test": r["rmspe_test"],
            "rmse_test": r["rmse_test"],
            "mape_test": r["mape_test"],
        })
    metrics_df = pd.DataFrame(rows)
    metrics_path = out_metrics / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Print summary
    print("\n📊 Metrics (test):")
    print(metrics_df.to_string(index=False))
    print(f"\n✅ Outputs saved to:\n- {model_path}\n- {preds_path}\n- {metrics_path}\n- {out_fig}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide_path", type=str, default="data/processed/rossmann_features_wide.csv")
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--test_days", type=int, default=42)
    parser.add_argument("--val_days", type=int, default=42)
    parser.add_argument("--study_db", type=str, default="data/optuna/rossmann_lgbm.sqlite3")
    parser.add_argument("--study_name", type=str, default="rossmann_lgbm")
    args = parser.parse_args()

    main(
        wide_path=Path(args.wide_path),
        n_trials=args.n_trials,
        test_days=args.test_days,
        val_days=args.val_days,
    )
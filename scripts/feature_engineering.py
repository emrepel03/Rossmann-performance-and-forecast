

# feature_engineering.py
# -------------------------------------------------------------
# Purpose: Build a clean, feature-rich dataset for modeling the
#          Rossmann Store Sales problem (daily sales forecasting).
#
# What it does:
#   1) Loads raw CSVs from data/raw/rossmann-store-sales/
#   2) Merges train.csv + store.csv
#   3) Cleans & imputes missing values (competition/promo fields)
#   4) Engineers features based on EDA findings
#   5) One-hot encodes categoricals
#   6) Creates lag/rolling time-series features per store
#   7) Saves:
#       - data/processed/rossmann_features_wide.csv  (numeric + one-hot)
#       - data/interim/rossmann_features_long.parquet (convenient for Python)
#       - data/processed/feature_dictionary.csv (what each feature means)
#
# Usage (from project root):
#   python scripts/feature_engineering.py
#   python scripts/feature_engineering.py --keep_closed  # keep closed days
#   python scripts/feature_engineering.py --no_ohe       # skip one-hot
#   python scripts/feature_engineering.py --raw_dir data/raw/rossmann-store-sales --out_dir data/processed
#
# -------------------------------------------------------------

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------
# Helper functions
# ------------------------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def iso_week_start_date(year: int, week: int) -> pd.Timestamp:
    """Return the Monday date for an ISO year/week (week starts Monday)."""
    return pd.to_datetime(f"{int(year)}-W{int(week):02d}-1", format="%G-W%V-%u", errors="coerce")


def months_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    """Approx months between two dates (signed)."""
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    return (d1.year - d2.year) * 12 + (d1.month - d2.month) + (d1.day - d2.day) / 30.0


def add_cyclical_encoding(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def make_lag_roll(df: pd.DataFrame, group_col: str, target_col: str,
                  lags=(7, 14, 28), rolls=(7, 14, 28)) -> pd.DataFrame:
    """Create lag and rolling-mean features per group (e.g., per Store)."""
    df = df.sort_values([group_col, "Date"]).copy()
    for L in lags:
        df[f"{target_col}_lag{L}"] = df.groupby(group_col)[target_col].shift(L)
    for R in rolls:
        df[f"{target_col}_roll{R}"] = (
            df.groupby(group_col)[target_col]
              .shift(1)  # avoid leakage of current day
              .rolling(R, min_periods=max(2, R//2))
              .mean()
        )
    return df


def one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=True)


# ------------------------
# Core pipeline
# ------------------------

def build_features(raw_dir: Path, out_dir: Path, keep_closed: bool = False, do_ohe: bool = True) -> None:
    interim_dir = Path("data/interim")
    ensure_dirs(out_dir, interim_dir)

    # 1) Load
    train = pd.read_csv(raw_dir / "train.csv", low_memory=False, dtype={"StateHoliday": "string"})
    store = pd.read_csv(raw_dir / "store.csv", low_memory=False)

    # 2) Merge & basic parsing
    df = train.merge(store, on="Store", how="left")
    df["Date"] = pd.to_datetime(df["Date"])  # YYYY-MM-DD
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # Optional: drop closed days for modeling stability (common Rossmann trick)
    if not keep_closed:
        df = df[df["Open"] == 1].copy()

    # 3) Basic date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1  # 1=Mon..7=Sun
    df["IsWeekend"] = (df["DayOfWeek"].isin([6, 7])).astype(int)
    df["IsDecember"] = (df["Month"] == 12).astype(int)
    df["DaysSinceStart"] = (df["Date"] - df["Date"].min()).dt.days

    # cyclical encodings (seasonality)
    df = add_cyclical_encoding(df, "Month", 12)
    df = add_cyclical_encoding(df, "DayOfWeek", 7)

    # === Promo_running: consecutive promo days so far (per store) ===
    # Vectorized version (avoids groupby.apply deprecation warnings)
    promo_int = df["Promo"].fillna(0).astype(int)
    # This key increases by 1 each time a zero appears within a store,
    # which resets the cumulative sum and yields consecutive run lengths.
    reset_key = promo_int.eq(0).groupby(df["Store"]).cumsum()
    df["Promo_running"] = promo_int.groupby([df["Store"], reset_key]).cumsum()

    # === StoreType × Month interaction (captures seasonal differences by type) ===
    _st_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    df["StoreType_code"] = df["StoreType"].map(_st_map)
    df["StoreType_x_Month"] = df["StoreType_code"].fillna(0) * df["Month"]

    # 4) Competition fields
    comp_med_by_type = df.groupby("StoreType")["CompetitionDistance"].median()
    global_med = df["CompetitionDistance"].median()
    def _fill_comp(row):
        if pd.notna(row["CompetitionDistance"]):
            return row["CompetitionDistance"]
        st = row["StoreType"]
        return comp_med_by_type.get(st, global_med)
    df["CompetitionDistance"] = df.apply(_fill_comp, axis=1)
    df["log_CompetitionDistance"] = np.log1p(df["CompetitionDistance"])
    df["HasCompetition"] = (df["CompetitionDistance"] > 0).astype(int)

    comp_year = df["CompetitionOpenSinceYear"].fillna(0).astype(int)
    comp_month = df["CompetitionOpenSinceMonth"].fillna(0).astype(int)
    comp_open_date = pd.to_datetime(
        dict(year=comp_year.where(comp_year>0, 1970), month=comp_month.where(comp_month>0, 1), day=1),
        errors="coerce"
    )
    df["MonthsSinceCompOpen"] = [
        months_between(cur, comp) if pd.notna(comp) else np.nan
        for cur, comp in zip(df["Date"], comp_open_date)
    ]
    df["MonthsSinceCompOpen"] = df["MonthsSinceCompOpen"].fillna(0).clip(lower=0)

    # 5) Promo2 features
    promo2_flag = df["Promo2"].fillna(0).astype(int)
    p2_year = df["Promo2SinceYear"].fillna(0).astype(int)
    p2_week = df["Promo2SinceWeek"].fillna(0).astype(int)
    p2_start = [
        iso_week_start_date(y, w) if (y>0 and w>0) else pd.NaT
        for y, w in zip(p2_year, p2_week)
    ]
    p2_start = pd.to_datetime(p2_start)
    df["WeeksSincePromo2Start"] = [
        ( (cur - start).days // 7 ) if (pd.notna(start) and (cur >= start)) else 0
        for cur, start in zip(df["Date"], p2_start)
    ]
    df["Promo2Running"] = ((promo2_flag==1) & (df["WeeksSincePromo2Start"]>0)).astype(int)

    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    # 6) Holidays features
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    df["AnyHoliday"] = ((df["StateHoliday"] != "0") | (df["SchoolHoliday"] == 1)).astype(int)

    # 7) Interaction features from EDA
    df["Promo_Month"] = df["Promo"] * df["Month"]
    st_map = {"a":0, "b":1, "c":2, "d":3}
    df["StoreType_code"] = df["StoreType"].map(st_map)
    df["Promo_StoreType"] = df["Promo"] * df["StoreType_code"].fillna(0)

    # 8) Lag & rolling features on Sales per Store
    df = make_lag_roll(df, group_col="Store", target_col="Sales",
                       lags=(7, 14, 28), rolls=(7, 14, 28))

    # Optional: clip extreme outliers on Sales/Customers
    for col, q in [("Sales", 0.99), ("Customers", 0.99) ]:
        cap = df[col].quantile(q)
        df[col] = np.where(df[col] > cap, cap, df[col])

    # 9) One-hot encoding (optional)
    long_out = df.copy()
    if do_ohe:
        cat_cols = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]
        df = one_hot(df, cat_cols)

    # 10) Save outputs
    ensure_dirs(Path("data/processed"))

    wide_path = Path("data/processed/rossmann_features_wide.csv")
    df.to_csv(wide_path, index=False)

    long_path = Path("data/interim/rossmann_features_long.parquet")
    long_out.to_parquet(long_path, index=False)

    # Feature dictionary
    feat_dict = [
        {"feature": "Sales", "description": "Target: daily store sales (clipped at 99th pct)"},
        {"feature": "Customers", "description": "Daily customers (clipped at 99th pct)"},
        {"feature": "Open", "description": "Store open flag (1/0)"},
        {"feature": "Promo", "description": "Promotion flag (1/0)"},
        {"feature": "SchoolHoliday", "description": "School holiday flag (1/0)"},
        {"feature": "StateHoliday", "description": "State holiday raw label (0/a/b/c)"},
        {"feature": "CompetitionDistance", "description": "Distance to nearest competitor (m)"},
        {"feature": "MonthsSinceCompOpen", "description": "Months since competitor opened (0 if unknown)"},
        {"feature": "Promo2", "description": "Continuous promotion program flag"},
        {"feature": "WeeksSincePromo2Start", "description": "Weeks since Promo2 start (0 if before/unknown)"},
        {"feature": "Promo2Running", "description": "Promo2 active and started (1/0)"},
        {"feature": "Month_sin/cos, DayOfWeek_sin/cos", "description": "Cyclical encodings for seasonality"},
        {"feature": "IsWeekend", "description": "1 if Saturday/Sunday"},
        {"feature": "IsDecember", "description": "1 if month==12"},
        {"feature": "DaysSinceStart", "description": "Days since first date in dataset"},
        {"feature": "Promo_Month", "description": "Interaction: Promo * Month"},
        {"feature": "Promo_StoreType", "description": "Interaction: Promo * StoreType code"},
        {"feature": "Promo_running", "description": "Consecutive promo days so far per store (no leakage)"},
        {"feature": "StoreType_x_Month", "description": "Interaction: StoreType code × Month (seasonal differences by type)"},
        {"feature": "Sales_lag*, Sales_roll*", "description": "Lagged sales & rolling means per store"}
    ]
    feat_df = pd.DataFrame(feat_dict)
    feat_df.to_csv(Path("data/processed/feature_dictionary.csv"), index=False)

    print("✅ Saved:")
    print(f" - {wide_path}")
    print(f" - {long_path}")
    print(" - data/processed/feature_dictionary.csv")


# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw/rossmann-store-sales",
                        help="Path to raw directory containing train.csv and store.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed",
                        help="Output directory for processed files")
    parser.add_argument("--keep_closed", action="store_true",
                        help="Keep closed days as training rows (default: drop)")
    parser.add_argument("--no_ohe", action="store_true",
                        help="Disable one-hot encoding of categorical columns")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    build_features(raw_dir=raw_dir, out_dir=out_dir,
                   keep_closed=args.keep_closed, do_ohe=(not args.no_ohe))
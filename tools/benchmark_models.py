"""Benchmark the learned cooldown-endpoint model against trivial baselines.

Reproduces the results table in the README. Leave-one-out cross-validation is
used because the dataset is small (n ~ 32); a single train/test split at this
size gives a test error dominated by which handful of junctions landed in the
test fold.

Usage
-----
    python tools/benchmark_models.py --data /path/to/unified/runs
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# No dataset ships with this repository. Point --data at your own runs.
DEFAULT_DATA = os.environ.get("ABAA_BENCHMARK_DATA", "")

# Features as the current trainer builds them. Kept verbatim so the benchmark
# measures the model that actually ships, not an idealised version of it.
FEATURE_ORDER = [
    "initial_drop_pct",
    "initial_drop_pct_5x",
    "initial_drop_pct_10x",
    "temperature",
    "voltage",
    "frequency",
    "initial_resistance",
    "first_baking_resistance",
    "jump_absolute",
    "jump_pct",
    "jump_absolute_weighted_3x",
    "baking_time",
    "last_baking_resistance",
    "baking_slope",
    "mean_rate",
    "second_derivative",
]
# Note: the extractor computes 18 features, but `train_rf_model` drops any
# column whose name contains "change" (baking_change, baking_pct_change), so
# 16 actually reach the model. That name-matching rule is fragile and is on
# the roadmap to be replaced by an explicit list.


def extract(path: str) -> dict | None:
    """Pull one feature row out of a unified run CSV. Returns None if unusable."""
    df = pd.read_csv(path)
    cooled = df["Cooled_Average_Resistance"].iloc[0]
    initial = df["Initial_Resistance (Ohms)"].iloc[0]
    if pd.isna(cooled) or pd.isna(initial):
        return None

    t = df["Time (s)"].to_numpy()
    r = df["Resistance (Ohms)"].to_numpy()
    keep = ~np.isnan(r)
    t, r = t[keep], r[keep]
    if len(r) < 10:
        return None

    first, last = r[0], r[-1]
    drop_pct = (first - initial) / initial * 100.0
    slope = stats.linregress(t[:10], r[:10]).slope
    first_deriv = np.diff(r[:10]) / np.maximum(np.diff(t[:10]), 1e-3)
    rate = float(np.mean(first_deriv))
    midpoints = (t[1:10] + t[:9]) / 2.0
    second_deriv = float(np.mean(np.diff(first_deriv) / np.maximum(np.diff(midpoints), 1e-3)))

    return {
        "initial_drop_pct": drop_pct,
        "initial_drop_pct_5x": drop_pct * 5.0,
        "initial_drop_pct_10x": drop_pct * 10.0,
        "temperature": 100.0,
        "voltage": 0.8,
        "frequency": 1.0,
        "initial_resistance": initial,
        "first_baking_resistance": first,
        "jump_absolute": first - initial,
        "jump_pct": drop_pct,
        "jump_absolute_weighted_3x": (first - initial) * 3.0,
        "baking_time": t[-1] - t[0],
        "last_baking_resistance": last,
        "baking_slope": slope,
        "mean_rate": rate,
        "second_derivative": second_deriv,
        "cooled_pct_change": (cooled - last) / last * 100.0,
    }


def load(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        sys.exit(f"No CSV files found in {data_dir!r}")
    rows = [row for row in (extract(p) for p in paths) if row is not None]
    if not rows:
        sys.exit(f"No usable runs in {data_dir!r}")
    return pd.DataFrame(rows)


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA, help="directory of unified run CSVs")
    args = ap.parse_args()

    if not args.data:
        sys.exit(
            "No data directory given.\n"
            "  python tools/benchmark_models.py --data /path/to/unified/runs\n"
            "or set ABAA_BENCHMARK_DATA. Measurement data is not distributed "
            "with this repository."
        )
    df = load(args.data)
    y = df.pop("cooled_pct_change").to_numpy()
    X = df[FEATURE_ORDER]
    drop_mag = X["initial_drop_pct"].abs().to_frame()
    loo = LeaveOneOut()

    print(f"\nLeave-one-out CV | n = {len(y)} junctions | {len(FEATURE_ORDER)} features")
    print(f"target: cooled_pct_change (mean {y.mean():.2f} %, std {y.std():.2f} %)\n")
    print(f"{'model':<34}{'RMSE %':>9}{'MAE %':>9}")
    print("-" * 52)

    rmse, mae = scores(y, np.full_like(y, y.mean()))
    print(f"{'predict the training mean':<34}{rmse:>9.2f}{mae:>9.2f}")

    candidates = [
        (
            "Random Forest (all features)",
            Pipeline([
                ("scale", StandardScaler()),
                ("rf", RandomForestRegressor(
                    n_estimators=50, max_depth=8, min_samples_leaf=2,
                    random_state=0, n_jobs=-1)),
            ]),
            X,
        ),
        (
            "Lasso (all features)",
            Pipeline([("scale", StandardScaler()), ("lasso", Lasso(alpha=0.5, max_iter=5000))]),
            X,
        ),
        (
            "OLS on |initial drop| only",
            Pipeline([("scale", StandardScaler()), ("ols", LinearRegression())]),
            drop_mag,
        ),
    ]
    for name, model, features in candidates:
        rmse, mae = scores(y, cross_val_predict(model, features, y, cv=loo))
        print(f"{name:<34}{rmse:>9.2f}{mae:>9.2f}")

    # One-parameter physical rule: rise = k * |initial drop|, k refit per fold.
    x = drop_mag.to_numpy().ravel()
    preds = np.array([
        np.mean(y[train] / x[train]) * x[test][0] for train, test in loo.split(x)
    ])
    rmse, mae = scores(y, preds)
    k_full = float(np.mean(y / x))
    print(f"{f'rule: rise = {k_full:.2f} x |drop|':<34}{rmse:>9.2f}{mae:>9.2f}")

    print(
        f"\ncorr(|initial drop|, rise) = {np.corrcoef(x, y)[0, 1]:.2f}"
        f"   |   in 1.5-2.3x band: "
        f"{int(((y >= 1.5 * x) & (y <= 2.3 * x)).sum())}/{len(y)}\n"
    )


if __name__ == "__main__":
    main()

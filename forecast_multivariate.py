import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

warnings.filterwarnings("ignore")

STORE_FILES = {
    "Baltimore": "baltimore.csv",
    "Lancaster": "lancaster.csv",
    "Philadelphia": "philadelphia.csv",
    "Richmond": "richmond.csv",
}

HORIZONS = (1, 2, 3, 4, 5, 6, 7)
N_BACKTEST_SPLITS = 12
SEASONAL_PERIOD = 7
MIN_TRAIN_SIZE = 180
MAX_H = max(HORIZONS)

# Lag sets: lags of Num_Customers + exogenous features (TouristEvent, Num_Employees,
# DayOfWeek of the forecast day)
LAG_SETS_MV = {
    "mv_lag7":  list(range(1, 8)),
    "mv_lag14": list(range(1, 15)),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_store_df(store_name: str, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Store"] = store_name
    df["TouristEvent_bin"] = (df["TouristEvent"] == "Yes").astype(float)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df


# ---------------------------------------------------------------------------
# Metrics  (same definitions as forecast_univariate.py)
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.mean(np.abs(y_true)))
    return float("nan") if denom == 0 else float(np.mean(np.abs(y_true - y_pred)) / denom)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _row_features(df: pd.DataFrame, t: int, lags: list[int], future_idx: int) -> list:
    """
    Feature vector to predict Num_Customers at future_idx, with last known index t.

    Lag features: Num_Customers at t, t-1, ..., t-(max_lag-1)
    Exogenous features (at forecast horizon):
        - TouristEvent (binary)       — assumed known in advance (planned event)
        - Num_Employees               — assumed known in advance (planned HR)
        - DayOfWeek of forecast day   — always known
    """
    lag_feats   = [float(df["Num_Customers"].iloc[t - (lag - 1)]) for lag in lags]
    tourist_h   = float(df["TouristEvent_bin"].iloc[future_idx])
    employees_h = float(df["Num_Employees"].iloc[future_idx])
    dow_h       = float(df["DayOfWeek"].iloc[future_idx])
    return lag_feats + [tourist_h, employees_h, dow_h]


def _build_train_dataset(
    df: pd.DataFrame, lags: list[int], h: int, up_to: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for direct forecasting at horizon h using df[:up_to]."""
    max_lag = max(lags)
    X, y = [], []
    for t in range(max_lag - 1, up_to - h):
        future_idx = t + h
        X.append(_row_features(df, t, lags, future_idx))
        y.append(float(df["Num_Customers"].iloc[future_idx]))
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _seasonal_naive_fallback(df: pd.DataFrame, origin: int) -> np.ndarray:
    """Seasonal naive: repeat the last 7 known values of Num_Customers."""
    base = df["Num_Customers"].iloc[origin - SEASONAL_PERIOD: origin].to_numpy(dtype=float)
    return np.array([base[(h - 1) % SEASONAL_PERIOD] for h in range(1, MAX_H + 1)], dtype=float)


def _future_pred_features(df: pd.DataFrame, t: int, h: int, lags: list[int]) -> np.ndarray:
    """
    Prediction features for a future horizon h beyond the end of df.
    Future exogenous variables are approximated by seasonal naive (same day 7 days earlier).
    """
    ref_idx     = max(0, t + h - SEASONAL_PERIOD)
    lag_feats   = [float(df["Num_Customers"].iloc[t - (lag - 1)]) for lag in lags]
    tourist_h   = float(df["TouristEvent_bin"].iloc[ref_idx])
    employees_h = float(df["Num_Employees"].iloc[ref_idx])
    dow_h       = float((df["Date"].iloc[t] + pd.Timedelta(days=h)).dayofweek)
    return np.array([lag_feats + [tourist_h, employees_h, dow_h]], dtype=float)


# ---------------------------------------------------------------------------
# ARIMAX forecaster
# ---------------------------------------------------------------------------

def _arimax_predict(df: pd.DataFrame, origin: int, for_future: bool = False) -> np.ndarray:
    """
    SARIMAX(1,1,1)(1,1,0,7) with exogenous regressors: TouristEvent_bin, Num_Employees, Pct_On_Sale.
    Seasonal order (1,1,0,7) captures the weekly pattern explicitly.
    """
    fallback = _seasonal_naive_fallback(df, origin)
    if origin < 2 * SEASONAL_PERIOD + 10:
        return fallback
    try:
        train_endog = df["Num_Customers"].iloc[:origin].to_numpy(dtype=float)
        exog_cols   = ["TouristEvent_bin", "Num_Employees", "Pct_On_Sale"]
        train_exog  = df[exog_cols].iloc[:origin].to_numpy(dtype=float)

        if for_future:
            ref_start   = max(0, origin - SEASONAL_PERIOD)
            future_exog = df[exog_cols].iloc[ref_start: ref_start + MAX_H].to_numpy(dtype=float)
        else:
            future_exog = df[exog_cols].iloc[origin: origin + MAX_H].to_numpy(dtype=float)

        if future_exog.shape[0] < MAX_H:
            return fallback

        fit = SARIMAX(
            train_endog,
            exog=train_exog,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, SEASONAL_PERIOD),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        pred = np.asarray(fit.forecast(steps=MAX_H, exog=future_exog), dtype=float)
        return pred
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# VAR forecaster
# ---------------------------------------------------------------------------

def _var_predict(df: pd.DataFrame, origin: int) -> np.ndarray:
    """
    VAR model on [Num_Customers, Sales]. Forecasts Num_Customers MAX_H steps ahead.
    Lag order selected automatically (max_lags=7) or falls back to lag=1.
    """
    fallback = _seasonal_naive_fallback(df, origin)
    if origin < 30:
        return fallback
    try:
        train = df[["Num_Customers", "Sales"]].iloc[:origin].to_numpy(dtype=float)
        model = VAR(train)
        # select lag order (up to 7, IC=aic) but cap to avoid overfitting
        try:
            lag_order = model.select_order(maxlags=min(7, origin // 10)).aic
            lag_order = max(1, lag_order)
        except Exception:
            lag_order = 1
        fit   = model.fit(lag_order)
        fc    = fit.forecast(train[-lag_order:], steps=MAX_H)   # shape (MAX_H, 2)
        return np.asarray(fc[:, 0], dtype=float)                # column 0 = Num_Customers
    except Exception:
        return fallback


def _seasonal_naive_from_series(series: np.ndarray, origin: int) -> np.ndarray:
    base = series[origin - SEASONAL_PERIOD: origin]
    return np.array([base[(h - 1) % SEASONAL_PERIOD] for h in range(1, MAX_H + 1)], dtype=float)


def _var4_predict_all(customers_wide: pd.DataFrame, origin: int) -> dict[str, np.ndarray]:
    """
    Global VAR model using Num_Customers from all four stores jointly.
    Returns one MAX_H forecast vector per store.
    """
    store_cols = list(customers_wide.columns)
    if origin < 30:
        return {
            s: _seasonal_naive_from_series(customers_wide[s].to_numpy(dtype=float), origin)
            for s in store_cols
        }

    try:
        train = customers_wide[store_cols].iloc[:origin].to_numpy(dtype=float)
        model = VAR(train)
        try:
            lag_order = model.select_order(maxlags=min(7, origin // 10)).aic
            lag_order = max(1, lag_order)
        except Exception:
            lag_order = 1
        fit = model.fit(lag_order)
        fc = fit.forecast(train[-lag_order:], steps=MAX_H)  # shape (MAX_H, n_stores)
        return {
            s: np.asarray(fc[:, i], dtype=float)
            for i, s in enumerate(store_cols)
        }
    except Exception:
        return {
            s: _seasonal_naive_from_series(customers_wide[s].to_numpy(dtype=float), origin)
            for s in store_cols
        }


def _build_customers_wide(dfs_by_store: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build aligned Date index matrix: columns are stores, values are Num_Customers."""
    merged: pd.DataFrame | None = None
    for store in STORE_FILES.keys():
        df = dfs_by_store[store][["Date", "Num_Customers"]].copy()
        df = df.rename(columns={"Num_Customers": store})
        merged = df if merged is None else merged.merge(df, on="Date", how="inner")

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged[[*STORE_FILES.keys()]]


def build_global_var_maps(
    dfs_by_store: dict[str, pd.DataFrame]
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, np.ndarray]]:
    """
    Precompute global VAR predictions per backtest origin and for future horizon.
    Returns:
      - per_store_origin_preds[store][origin] = prediction vector
      - per_store_future_preds[store] = prediction vector at origin=len(series)
    """
    customers_wide = _build_customers_wide(dfs_by_store)
    origins = get_backtest_origins(len(customers_wide))

    per_store_origin_preds = {s: {} for s in STORE_FILES.keys()}
    for origin in origins:
        preds_all = _var4_predict_all(customers_wide, origin)
        for s in STORE_FILES.keys():
            per_store_origin_preds[s][origin] = preds_all[s]

    per_store_future_preds = _var4_predict_all(customers_wide, len(customers_wide))
    return per_store_origin_preds, per_store_future_preds


# ---------------------------------------------------------------------------
# Linear Regression MV (direct multi-step, same lag features as RF/XGBoost)
# ---------------------------------------------------------------------------

def _linreg_mv_predict(
    df: pd.DataFrame, origin: int, lags: list[int], for_future: bool = False
) -> np.ndarray:
    """
    Linear Regression with the same lag+exogenous feature set as RF/XGBoost MV.
    StandardScaler applied inside to keep coefficients stable.
    """
    fallback = _seasonal_naive_fallback(df, origin)
    if origin < max(lags) + MAX_H + 10:
        return fallback
    try:
        t = origin - 1
        preds = []
        for h in range(1, MAX_H + 1):
            X_tr, y_tr = _build_train_dataset(df, lags, h, origin)
            if len(X_tr) < 10:
                preds.append(fallback[h - 1])
                continue
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            model  = LinearRegression()
            model.fit(X_tr_s, y_tr)
            if for_future:
                X_pred = _future_pred_features(df, t, h, lags)
            else:
                X_pred = np.array([_row_features(df, t, lags, origin + h - 1)], dtype=float)
            X_pred_s = scaler.transform(X_pred)
            preds.append(float(model.predict(X_pred_s)[0]))
        return np.array(preds, dtype=float)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# ML multivariate forecasters  (direct multi-step strategy)
# ---------------------------------------------------------------------------

def _rf_mv_predict(
    df: pd.DataFrame, origin: int, lags: list[int], for_future: bool = False
) -> np.ndarray:
    """
    Train RandomForest on df[:origin], return MAX_H predictions.
    for_future=False: backtest — actual future exogenous values available in df.
    for_future=True : real forecast — future exogenous approximated by seasonal naive.
    """
    fallback = _seasonal_naive_fallback(df, origin)
    if origin < max(lags) + MAX_H + 10:
        return fallback
    try:
        t = origin - 1
        preds = []
        for h in range(1, MAX_H + 1):
            X_tr, y_tr = _build_train_dataset(df, lags, h, origin)
            if len(X_tr) < 10:
                preds.append(fallback[h - 1])
                continue
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            if for_future:
                X_pred = _future_pred_features(df, t, h, lags)
            else:
                X_pred = np.array([_row_features(df, t, lags, origin + h - 1)], dtype=float)
            preds.append(float(model.predict(X_pred)[0]))
        return np.array(preds, dtype=float)
    except Exception:
        return fallback


def _xgb_mv_predict(
    df: pd.DataFrame, origin: int, lags: list[int], for_future: bool = False
) -> np.ndarray:
    """
    Train XGBoost on df[:origin], return MAX_H predictions.
    for_future=False: backtest — actual future exogenous values available in df.
    for_future=True : real forecast — future exogenous approximated by seasonal naive.
    """
    fallback = _seasonal_naive_fallback(df, origin)
    if origin < max(lags) + MAX_H + 10:
        return fallback
    try:
        t = origin - 1
        preds = []
        for h in range(1, MAX_H + 1):
            X_tr, y_tr = _build_train_dataset(df, lags, h, origin)
            if len(X_tr) < 10:
                preds.append(fallback[h - 1])
                continue
            model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            model.fit(X_tr, y_tr)
            if for_future:
                X_pred = _future_pred_features(df, t, h, lags)
            else:
                X_pred = np.array([_row_features(df, t, lags, origin + h - 1)], dtype=float)
            preds.append(float(model.predict(X_pred)[0]))
        return np.array(preds, dtype=float)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def get_backtest_origins(n: int) -> list[int]:
    latest_origin = n - MAX_H
    first_origin  = latest_origin - N_BACKTEST_SPLITS + 1
    if first_origin < MIN_TRAIN_SIZE:
        raise ValueError(
            f"Not enough data: first_origin={first_origin} < {MIN_TRAIN_SIZE}"
        )
    return list(range(first_origin, latest_origin + 1))


def run_mv_backtest(
    df: pd.DataFrame,
    var_preds_by_origin: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    rows = []
    series  = df["Num_Customers"].to_numpy(dtype=float)
    origins = get_backtest_origins(len(df))

    for split_id, origin in enumerate(origins, start=1):
        y_future = series[origin: origin + MAX_H]

        # --- ARIMAX and VAR (no lag set: fixed "-") ---
        var_pred = (
            var_preds_by_origin[origin]
            if var_preds_by_origin is not None and origin in var_preds_by_origin
            else _var_predict(df, origin)
        )
        fixed_preds = [
            ("ARIMAX", "-", _arimax_predict(df, origin, for_future=False)),
            ("VAR",    "-", var_pred),
        ]
        for method_name, lag_name, pred_vec in fixed_preds:
            for h in HORIZONS:
                rows.append({
                    "Split":   split_id,
                    "Horizon": h,
                    "Method":  method_name,
                    "LagSet":  lag_name,
                    "y_true":  float(y_future[h - 1]),
                    "y_pred":  float(pred_vec[h - 1]),
                })

        # --- lag-based ML methods ---
        for lag_name, lags in LAG_SETS_MV.items():
            mv_preds = [
                ("RandomForest_MV",  _rf_mv_predict(df, origin, lags, for_future=False)),
                ("XGBoost_MV",       _xgb_mv_predict(df, origin, lags, for_future=False)),
                ("LinearReg_MV",     _linreg_mv_predict(df, origin, lags, for_future=False)),
            ]
            for method_name, pred_vec in mv_preds:
                for h in HORIZONS:
                    rows.append({
                        "Split":   split_id,
                        "Horizon": h,
                        "Method":  method_name,
                        "LagSet":  lag_name,
                        "y_true":  float(y_future[h - 1]),
                        "y_pred":  float(pred_vec[h - 1]),
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric aggregation  (median per split — same approach as univariate)
# ---------------------------------------------------------------------------

def aggregate_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    # 1. per-split metrics
    split_rows = []
    for keys, grp in pred_df.groupby(["Store", "Method", "LagSet", "Horizon", "Split"]):
        y_true = grp["y_true"].to_numpy(dtype=float)
        y_pred = grp["y_pred"].to_numpy(dtype=float)
        split_rows.append({
            "Store":   keys[0],
            "Method":  keys[1],
            "LagSet":  keys[2],
            "Horizon": keys[3],
            "Split":   keys[4],
            "MAE":  mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "NMAE": nmae(y_true, y_pred),
        })
    split_df = pd.DataFrame(split_rows)

    # 2. median aggregation over splits
    agg = (
        split_df
        .groupby(["Store", "Method", "LagSet", "Horizon"])[["MAE", "RMSE", "NMAE"]]
        .median().round(3).reset_index()
    )
    counts = (
        split_df
        .groupby(["Store", "Method", "LagSet", "Horizon"])["Split"]
        .count().reset_index(name="Splits")
    )
    out = counts.merge(agg, on=["Store", "Method", "LagSet", "Horizon"])

    return out.sort_values(["Store", "Horizon", "NMAE", "MAE"]).reset_index(drop=True)


def add_improvement_vs_seasonal(
    mv_agg: pd.DataFrame, univ_metrics_path: Path
) -> pd.DataFrame:
    """
    Improvement % vs SeasonalNaive7 baseline (NMAE from univariate results).
    Keeps the same formula as forecast_univariate.py.
    """
    out = mv_agg.copy()
    out["Improvement_vs_SeasonalNaive7_pct"] = np.nan

    if not univ_metrics_path.exists():
        return out

    univ = pd.read_csv(univ_metrics_path)
    baseline_df = univ[(univ["Method"] == "SeasonalNaive7") & (univ["LagSet"] == "-")]

    for (store, horizon), grp in out.groupby(["Store", "Horizon"]):
        baseline = baseline_df[
            (baseline_df["Store"] == store) & (baseline_df["Horizon"] == horizon)
        ]
        if baseline.empty:
            continue
        baseline_nmae = float(baseline.iloc[0]["NMAE"])
        if baseline_nmae == 0:
            continue
        idx = (out["Store"] == store) & (out["Horizon"] == horizon)
        out.loc[idx, "Improvement_vs_SeasonalNaive7_pct"] = (
            (baseline_nmae - out.loc[idx, "NMAE"]) / baseline_nmae
        ) * 100

    out["Improvement_vs_SeasonalNaive7_pct"] = (
        out["Improvement_vs_SeasonalNaive7_pct"].round(3)
    )
    return out


def choose_best_method(agg: pd.DataFrame) -> pd.DataFrame:
    return (
        agg.sort_values(["Store", "Horizon", "NMAE", "MAE"])
        .groupby(["Store", "Horizon"]).head(1)
        .reset_index(drop=True)
    )


def forecast_next7_mv(
    df: pd.DataFrame,
    method: str,
    lag_set: str,
    store_name: str | None = None,
    var_future_preds: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if method == "ARIMAX":
        return _arimax_predict(df, len(df), for_future=True)
    if method == "VAR":
        if store_name is not None and var_future_preds is not None and store_name in var_future_preds:
            return var_future_preds[store_name]
        return _var_predict(df, len(df))
    lags = LAG_SETS_MV[lag_set]
    if method == "RandomForest_MV":
        return _rf_mv_predict(df, len(df), lags, for_future=True)
    if method == "XGBoost_MV":
        return _xgb_mv_predict(df, len(df), lags, for_future=True)
    if method == "LinearReg_MV":
        return _linreg_mv_predict(df, len(df), lags, for_future=True)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_predictions = []
    dfs_by_store: dict[str, pd.DataFrame] = {}

    print("=" * 90)
    print(" MULTIVARIATE FORECASTING - ARIMAX, VAR(4 stores), LinearReg MV, RandomForest MV, XGBoost MV")
    print("=" * 90)
    print(f"Backtesting rolling splits : {N_BACKTEST_SPLITS}")
    print("Extra features             : TouristEvent, Num_Employees, DayOfWeek\n")

    for store_name, file_name in STORE_FILES.items():
        df = load_store_df(store_name, file_name)
        dfs_by_store[store_name] = df
        print(
            f"[{store_name}] rows={len(df)} | "
            f"dates={df['Date'].min().date()} to {df['Date'].max().date()}"
        )

    global_var_backtest, global_var_future = build_global_var_maps(dfs_by_store)

    for store_name, df in dfs_by_store.items():
        pred_df = run_mv_backtest(df, var_preds_by_origin=global_var_backtest.get(store_name))
        pred_df["Store"] = store_name
        all_predictions.append(pred_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = aggregate_metrics(predictions_df)
    metrics_df = add_improvement_vs_seasonal(
        metrics_df, Path("univariate_metrics_summary.csv")
    )
    best_df = choose_best_method(metrics_df)

    # Next 7-day forecast using best H=7 method
    next7_rows = []
    for store_name, df in dfs_by_store.items():
        best_h7 = best_df[(best_df["Store"] == store_name) & (best_df["Horizon"] == 7)]
        chosen_method = str(best_h7.iloc[0]["Method"]) if not best_h7.empty else "RandomForest_MV"
        chosen_lagset = str(best_h7.iloc[0]["LagSet"]) if not best_h7.empty else "mv_lag7"

        preds    = forecast_next7_mv(
            df,
            chosen_method,
            chosen_lagset,
            store_name=store_name,
            var_future_preds=global_var_future,
        )
        last_date = df["Date"].max()

        for h in HORIZONS:
            next7_rows.append({
                "Store":               store_name,
                "ChosenMethod_H7":     chosen_method,
                "ChosenLagSet_H7":     chosen_lagset,
                "Horizon":             h,
                "ForecastDate":        (last_date + pd.Timedelta(days=h)).date(),
                "Pred_Num_Customers":  round(max(float(preds[h - 1]), 0.0), 2),
            })

    next7_df = pd.DataFrame(next7_rows)

    out_predictions = Path("multivariate_backtest_all_splits.csv")
    out_metrics     = Path("multivariate_metrics_summary.csv")
    out_best        = Path("multivariate_best_methods.csv")
    out_next7       = Path("multivariate_next7.csv")

    predictions_df.to_csv(out_predictions, index=False)
    metrics_df.to_csv(out_metrics, index=False)
    best_df.to_csv(out_best, index=False)
    next7_df.to_csv(out_next7, index=False)

    print("\nBest MV method by store and horizon (NMAE then MAE):")
    display_cols = ["Store", "Horizon", "Method", "LagSet", "NMAE", "MAE"]
    if "Improvement_vs_SeasonalNaive7_pct" in best_df.columns:
        display_cols.append("Improvement_vs_SeasonalNaive7_pct")
    print(best_df[display_cols])

    print("\nNext 7-day forecasts by chosen H=7 MV strategy:")
    print(next7_df)

    print("\nGenerated files:")
    for p in [out_predictions, out_metrics, out_best, out_next7]:
        print(p.resolve())


if __name__ == "__main__":
    main()

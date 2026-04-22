import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


warnings.filterwarnings("ignore")

STORE_FILES = {
    "Baltimore": "csv/stores/baltimore.csv",
    "Lancaster": "csv/stores/lancaster.csv",
    "Philadelphia": "csv/stores/philadelphia.csv",
    "Richmond": "csv/stores/richmond.csv",
}
METHODS = ["SeasonalNaive7", "ETS_HoltWinters", "ARIMA", "RandomForest", "XGBoost"]
HORIZONS = (1, 2, 3, 4, 5, 6, 7)
N_BACKTEST_SPLITS = 12
SEASONAL_PERIOD = 7
MIN_TRAIN_SIZE = 180
MAX_H = max(HORIZONS)

# Conjuntos de lags a testar para os métodos ML
LAG_SETS = {
    "lag7":  list(range(1, 8)),   # lags 1-7
    "lag14": list(range(1, 15)),  # lags 1-14
}


def load_store_series(store_name: str, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Store"] = store_name
    return df[["Date", "Num_Customers", "Store"]]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.mean(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def seasonal_naive_forecast(train: np.ndarray, max_h: int) -> np.ndarray:
    if len(train) < SEASONAL_PERIOD:
        return np.repeat(float(train[-1]), max_h)
    base = train[-SEASONAL_PERIOD:]
    return np.array([base[(h - 1) % SEASONAL_PERIOD] for h in range(1, max_h + 1)], dtype=float)

# ETS/Holt-Winters com aditivo para tendência e sazonalidade, e período sazonal de 7 dias. Se o ajuste falhar ou se a série for muito curta, retorna a previsão da Seasonal Naive.
def ets_forecast(train: np.ndarray, max_h: int) -> np.ndarray:
    fallback = seasonal_naive_forecast(train, max_h)
    if len(train) < (2 * SEASONAL_PERIOD):
        return fallback

    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=False)
        pred = np.asarray(fit.forecast(max_h), dtype=float)
        return pred
    except Exception:
        return fallback


def arima_forecast(train: np.ndarray, max_h: int) -> np.ndarray:
    fallback = seasonal_naive_forecast(train, max_h)
    if len(train) < 40:
        return fallback

    try:
        fit = ARIMA(
            train,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        pred = np.asarray(fit.forecast(steps=max_h), dtype=float)
        return pred
    except Exception:
        return fallback


def _build_direct_dataset(train: np.ndarray, lags: list[int], h: int) -> tuple[np.ndarray, np.ndarray]:
    """Constrói dataset para direct forecasting: features = lags em t, target = t+h."""
    max_lag = max(lags)
    X, y = [], []
    for t in range(max_lag - 1, len(train) - h):
        X.append([train[t - (lag - 1)] for lag in lags])
        y.append(train[t + h])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def _predict_features(train: np.ndarray, lags: list[int]) -> np.ndarray:
    return np.array([[train[-lag] for lag in lags]], dtype=float)


def rf_forecast(train: np.ndarray, max_h: int, lags: list[int]) -> np.ndarray:
    fallback = seasonal_naive_forecast(train, max_h)
    if len(train) < max(lags) + max_h + 10:
        return fallback
    try:
        X_pred = _predict_features(train, lags)
        preds = []
        for h in range(1, max_h + 1):
            X_tr, y_tr = _build_direct_dataset(train, lags, h)
            if len(X_tr) < 10:
                preds.append(fallback[h - 1])
                continue
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds.append(float(model.predict(X_pred)[0]))
        return np.array(preds, dtype=float)
    except Exception:
        return fallback


def xgb_forecast(train: np.ndarray, max_h: int, lags: list[int]) -> np.ndarray:
    fallback = seasonal_naive_forecast(train, max_h)
    if len(train) < max(lags) + max_h + 10:
        return fallback
    try:
        X_pred = _predict_features(train, lags)
        preds = []
        for h in range(1, max_h + 1):
            X_tr, y_tr = _build_direct_dataset(train, lags, h)
            if len(X_tr) < 10:
                preds.append(fallback[h - 1])
                continue
            model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            model.fit(X_tr, y_tr)
            preds.append(float(model.predict(X_pred)[0]))
        return np.array(preds, dtype=float)
    except Exception:
        return fallback


def get_backtest_origins(n: int) -> list[int]:
    latest_origin = n - MAX_H
    first_origin = latest_origin - N_BACKTEST_SPLITS + 1

    if first_origin < MIN_TRAIN_SIZE:
        raise ValueError(
            f"Not enough data for rolling backtest. first_origin={first_origin} < {MIN_TRAIN_SIZE}."
        )

    return list(range(first_origin, latest_origin + 1))


def run_rolling_backtest(series: np.ndarray) -> pd.DataFrame:
    rows = []
    origins = get_backtest_origins(len(series))

    for split_id, origin in enumerate(origins, start=1):
        train = series[:origin]
        y_future = series[origin:origin + MAX_H]

        pred_map = {
            "SeasonalNaive7": seasonal_naive_forecast(train, MAX_H),
            "ETS_HoltWinters": ets_forecast(train, MAX_H),
            "ARIMA": arima_forecast(train, MAX_H),
        }

        for method_name, pred_vec in pred_map.items():
            for h in HORIZONS:
                rows.append(
                    {
                        "Split": split_id,
                        "Horizon": h,
                        "Method": method_name,
                        "LagSet": "-",
                        "y_true": float(y_future[h - 1]),
                        "y_pred": float(pred_vec[h - 1]),
                    }
                )

        for lag_name, lags in LAG_SETS.items():
            ml_preds = [
                ("RandomForest", rf_forecast(train, MAX_H, lags)),
                ("XGBoost",      xgb_forecast(train, MAX_H, lags)),
            ]
            for ml_name, pred_vec in ml_preds:
                for h in HORIZONS:
                    rows.append(
                        {
                            "Split": split_id,
                            "Horizon": h,
                            "Method": ml_name,
                            "LagSet": lag_name,
                            "y_true": float(y_future[h - 1]),
                            "y_pred": float(pred_vec[h - 1]),
                        }
                    )

    return pd.DataFrame(rows)


def aggregate_metrics_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    # 1. calcular métricas por split individual (exceto R2, que precisa de múltiplos pontos)
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

    # 2. agregar via mediana (robusta a outliers) sobre os splits
    median_cols = ["MAE", "RMSE", "NMAE"]
    agg = (
        split_df
        .groupby(["Store", "Method", "LagSet", "Horizon"])[median_cols]
        .median()
        .round(3)
        .reset_index()
    )
    counts = (
        split_df
        .groupby(["Store", "Method", "LagSet", "Horizon"])["Split"]
        .count()
        .reset_index(name="Splits")
    )
    out = counts.merge(agg, on=["Store", "Method", "LagSet", "Horizon"])

    return out.sort_values(["Store", "Horizon", "NMAE", "MAE"]).reset_index(drop=True)


def add_improvement_vs_seasonal(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg.copy()
    out["Improvement_vs_SeasonalNaive7_pct"] = np.nan

    for (store, horizon), grp in out.groupby(["Store", "Horizon"]):
        baseline = grp[(grp["Method"] == "SeasonalNaive7") & (grp["LagSet"] == "-")]
        if baseline.empty:
            continue

        baseline_nmae = float(baseline["NMAE"].iloc[0])
        if baseline_nmae == 0:
            continue

        idx = (out["Store"] == store) & (out["Horizon"] == horizon)
        out.loc[idx, "Improvement_vs_SeasonalNaive7_pct"] = (
            (baseline_nmae - out.loc[idx, "NMAE"]) / baseline_nmae
        ) * 100

    out["Improvement_vs_SeasonalNaive7_pct"] = out["Improvement_vs_SeasonalNaive7_pct"].round(3)
    return out


def choose_best_method(agg: pd.DataFrame) -> pd.DataFrame:
    return (
        agg.sort_values(["Store", "Horizon", "NMAE", "MAE"]).groupby(["Store", "Horizon"]).head(1).reset_index(drop=True)
    )


def forecast_next7(train: np.ndarray, method: str, lag_set: str = "-") -> np.ndarray:
    if method == "SeasonalNaive7":
        return seasonal_naive_forecast(train, MAX_H)
    if method == "ETS_HoltWinters":
        return ets_forecast(train, MAX_H)
    if method == "ARIMA":
        return arima_forecast(train, MAX_H)
    if method == "RandomForest":
        return rf_forecast(train, MAX_H, LAG_SETS[lag_set])
    if method == "XGBoost":
        return xgb_forecast(train, MAX_H, LAG_SETS[lag_set])
    raise ValueError(f"Unknown method: {method}")


def main():
    all_predictions = []
    series_by_store = {}

    print("=" * 90)
    print(" UNIVARIATE FORECASTING - Seasonal Naive, Holt-Winters/ETS, ARIMA, Random Forest, XGBoost")
    print("=" * 90)
    print(f"Backtesting rolling splits: {N_BACKTEST_SPLITS}")

    for store_name, file_name in STORE_FILES.items():
        df = load_store_series(store_name, file_name)
        series = df["Num_Customers"].to_numpy(dtype=float)
        series_by_store[store_name] = (df, series)

        print(
            f"[{store_name}] rows={len(df)} | dates={df['Date'].min().date()} to {df['Date'].max().date()}"
        )

        pred_df = run_rolling_backtest(series)
        pred_df["Store"] = store_name
        all_predictions.append(pred_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = aggregate_metrics_from_predictions(predictions_df)
    metrics_df = add_improvement_vs_seasonal(metrics_df)
    best_df = choose_best_method(metrics_df)

    next7_rows = []
    for store_name, (df, series) in series_by_store.items():
        best_h7 = best_df[(best_df["Store"] == store_name) & (best_df["Horizon"] == 7)]

        if best_h7.empty:
            chosen_method = "SeasonalNaive7"
            chosen_lagset = "-"
        else:
            chosen_method = str(best_h7.iloc[0]["Method"])
            chosen_lagset = str(best_h7.iloc[0]["LagSet"])

        preds = forecast_next7(series, chosen_method, chosen_lagset)
        last_date = df["Date"].max()

        for h in HORIZONS:
            next7_rows.append(
                {
                    "Store": store_name,
                    "ChosenMethod_H7": chosen_method,
                    "ChosenLagSet_H7": chosen_lagset,
                    "Horizon": h,
                    "ForecastDate": (last_date + pd.Timedelta(days=h)).date(),
                    "Pred_Num_Customers": round(max(float(preds[h - 1]), 0.0), 2),
                }
            )

    next7_df = pd.DataFrame(next7_rows)

    out_predictions = Path("univariate_backtest_all_splits.csv")
    out_metrics = Path("univariate_metrics_summary.csv")
    out_best = Path("univariate_best_methods.csv")
    out_next7 = Path("univariate_next7.csv")

    predictions_df.to_csv(out_predictions, index=False)
    metrics_df.to_csv(out_metrics, index=False)
    best_df.to_csv(out_best, index=False)
    next7_df.to_csv(out_next7, index=False)

    print("\nBest method by store and horizon (NMAE then MAE):")
    print(best_df[["Store", "Horizon", "Method", "NMAE", "MAE", "Improvement_vs_SeasonalNaive7_pct"]])

    print("\nNext 7-day forecasts by chosen H=7 strategy:")
    print(next7_df)

    print("\nGenerated files:")
    print(out_predictions.resolve())
    print(out_metrics.resolve())
    print(out_best.resolve())
    print(out_next7.resolve())


if __name__ == "__main__":
    main()

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


warnings.filterwarnings("ignore")

STORE_FILES = {
    "Baltimore": "baltimore.csv",
    "Lancaster": "lancaster.csv",
    "Philadelphia": "philadelphia.csv",
    "Richmond": "richmond.csv",
}

METHODS = ["SeasonalNaive7", "ETS_HoltWinters", "ARIMA"]
HORIZONS = (1, 2, 3, 4, 5, 6, 7)
N_BACKTEST_SPLITS = 12
SEASONAL_PERIOD = 7
MIN_TRAIN_SIZE = 180
MAX_H = max(HORIZONS)


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


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(np.abs(y_true - y_pred) / denom) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1 - (ss_res / ss_tot)


def nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.mean(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAPE": round(mape(y_true, y_pred), 3),
        "sMAPE": round(smape(y_true, y_pred), 3),
        "R2": round(r2(y_true, y_pred), 4),
        "NMAE": round(nmae(y_true, y_pred), 4),
    }


def seasonal_naive_forecast(train: np.ndarray, max_h: int) -> np.ndarray:
    if len(train) < SEASONAL_PERIOD:
        return np.repeat(float(train[-1]), max_h)
    base = train[-SEASONAL_PERIOD:]
    return np.array([base[(h - 1) % SEASONAL_PERIOD] for h in range(1, max_h + 1)], dtype=float)


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

    return pd.DataFrame(rows)


def aggregate_metrics_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    metric_rows = []

    for keys, grp in pred_df.groupby(["Store", "Method", "LagSet", "Horizon"]):
        y_true = grp["y_true"].to_numpy(dtype=float)
        y_pred = grp["y_pred"].to_numpy(dtype=float)
        row = {
            "Store": keys[0],
            "Method": keys[1],
            "LagSet": keys[2],
            "Horizon": keys[3],
            "Splits": len(grp),
            **evaluate(y_true, y_pred),
        }
        metric_rows.append(row)

    out = pd.DataFrame(metric_rows)
    return out.sort_values(["Store", "Horizon", "sMAPE", "MAE"]).reset_index(drop=True)


def add_improvement_vs_seasonal(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg.copy()
    out["Improvement_vs_SeasonalNaive7_pct"] = np.nan

    for (store, horizon), grp in out.groupby(["Store", "Horizon"]):
        baseline = grp[(grp["Method"] == "SeasonalNaive7") & (grp["LagSet"] == "-")]
        if baseline.empty:
            continue

        baseline_smape = float(baseline["sMAPE"].iloc[0])
        if baseline_smape == 0:
            continue

        idx = (out["Store"] == store) & (out["Horizon"] == horizon)
        out.loc[idx, "Improvement_vs_SeasonalNaive7_pct"] = (
            (baseline_smape - out.loc[idx, "sMAPE"]) / baseline_smape
        ) * 100

    out["Improvement_vs_SeasonalNaive7_pct"] = out["Improvement_vs_SeasonalNaive7_pct"].round(3)
    return out


def choose_best_method(agg: pd.DataFrame) -> pd.DataFrame:
    return (
        agg.sort_values(["Store", "Horizon", "sMAPE", "MAE"]).groupby(["Store", "Horizon"]).head(1).reset_index(drop=True)
    )


def forecast_next7(train: np.ndarray, method: str) -> np.ndarray:
    if method == "SeasonalNaive7":
        return seasonal_naive_forecast(train, MAX_H)
    if method == "ETS_HoltWinters":
        return ets_forecast(train, MAX_H)
    if method == "ARIMA":
        return arima_forecast(train, MAX_H)
    raise ValueError(f"Unknown method: {method}")


def main():
    all_predictions = []
    series_by_store = {}

    print("=" * 90)
    print(" UNIVARIATE FORECASTING - Seasonal Naive, Holt-Winters/ETS, ARIMA")
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
        else:
            chosen_method = str(best_h7.iloc[0]["Method"])

        preds = forecast_next7(series, chosen_method)
        last_date = df["Date"].max()

        for h in HORIZONS:
            next7_rows.append(
                {
                    "Store": store_name,
                    "ChosenMethod_H7": chosen_method,
                    "ChosenLagSet_H7": "-",
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

    print("\nBest method by store and horizon (sMAPE then MAE):")
    print(best_df[["Store", "Horizon", "Method", "sMAPE", "MAE", "R2", "Improvement_vs_SeasonalNaive7_pct"]])

    print("\nNext 7-day forecasts by chosen H=7 strategy:")
    print(next7_df)

    print("\nGenerated files:")
    print(out_predictions.resolve())
    print(out_metrics.resolve())
    print(out_best.resolve())
    print(out_next7.resolve())


if __name__ == "__main__":
    main()

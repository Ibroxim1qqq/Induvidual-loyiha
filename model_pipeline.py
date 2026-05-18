from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Ba'zi Windows tizimlarida joblib fizik yadrolarni aniqlay olmaydi.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


MODEL_NAMES = [
    "Ridge Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "Extra Trees Regressor",
]
BASELINE_NAME = "Naive Baseline"
ENSEMBLE_NAME = "Mean Ensemble"
BLEND_NAME = "Conservative Blend"
DERIVED_FORECAST_NAMES = [ENSEMBLE_NAME, BLEND_NAME]
ALL_FORECAST_NAMES = [BASELINE_NAME, *MODEL_NAMES, *DERIVED_FORECAST_NAMES]

TEST_YEARS = 1
LAG_DAYS = 30
ROLLING_WINDOWS = (5, 10, 20, 30)
RETURN_PERIODS = (1, 5, 10, 20)
MOMENTUM_PERIODS = (5, 10, 20)
EMA_SPANS = (5, 10, 20, 30)
WEEKLY_RETURN_LAGS = 26
WEEKLY_WINDOWS = (4, 8, 13, 26)
ROLLING_BACKTEST_ORIGINS = 2
FEATURE_COLUMNS = [
    *[f"lag_{lag}" for lag in range(1, LAG_DAYS + 1)],
    *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
    *[f"rolling_std_{window}" for window in ROLLING_WINDOWS],
    *[f"return_{period}" for period in RETURN_PERIODS],
    *[f"momentum_{period}" for period in MOMENTUM_PERIODS],
    *[f"ema_{span}" for span in EMA_SPANS],
    "day_index",
]


def split_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tarixning oxirgi 1 yilini test, undan oldingisini train sifatida ajratadi."""
    cutoff_date = data.index.max() - pd.DateOffset(years=TEST_YEARS)
    train_data = data[data.index < cutoff_date].copy()
    test_data = data[data.index >= cutoff_date].copy()

    if len(train_data) <= LAG_DAYS + 30 or len(test_data) < 30:
        split_index = int(len(data) * 0.9)
        train_data = data.iloc[:split_index].copy()
        test_data = data.iloc[split_index:].copy()

    return train_data, test_data


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Kunlik modellar uchun lag, trend va volatilitet feature'larini yaratadi."""
    features = data.copy()

    for lag in range(1, LAG_DAYS + 1):
        features[f"lag_{lag}"] = features["Close"].shift(lag)

    for window in ROLLING_WINDOWS:
        features[f"rolling_mean_{window}"] = (
            features["Close"].shift(1).rolling(window).mean()
        )
        features[f"rolling_std_{window}"] = (
            features["Close"].shift(1).rolling(window).std()
        )

    for period in RETURN_PERIODS:
        features[f"return_{period}"] = features["Close"].pct_change(period).shift(1)

    for period in MOMENTUM_PERIODS:
        features[f"momentum_{period}"] = (
            features["Close"].shift(1) - features["Close"].shift(period + 1)
        )

    for span in EMA_SPANS:
        features[f"ema_{span}"] = features["Close"].shift(1).ewm(
            span=span,
            adjust=False,
        ).mean()

    features["day_index"] = np.arange(len(features))
    features["target_change"] = features["Close"].diff()
    return features.dropna()


def get_sklearn_model_builders() -> dict[str, Callable[[], object]]:
    """Ilovadagi asosiy kunlik ML modellarini yaratadi."""
    return {
        "Ridge Regression": lambda: Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=2.0))]
        ),
        "Random Forest Regressor": lambda: RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
        "Gradient Boosting Regressor": lambda: GradientBoostingRegressor(
            n_estimators=90,
            learning_rate=0.03,
            max_depth=2,
            random_state=42,
        ),
        "Extra Trees Regressor": lambda: ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
    }


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Model sifatini MAE, RMSE, MAPE va R2 orqali hisoblaydi."""
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    safe_actual = np.where(actual_values == 0, np.nan, actual_values)

    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "MAPE": float(
            np.nanmean(np.abs((actual_values - predicted_values) / safe_actual)) * 100
        ),
        "R2 Score": float(r2_score(actual_values, predicted_values)),
    }


def evaluate_models(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series], str]:
    """4 ta modelni oxirgi 1 yillik test davrida solishtiradi."""
    feature_frame = build_feature_frame(data)
    train_features = feature_frame.loc[feature_frame.index <= train_data.index.max()]
    test_features = feature_frame.loc[feature_frame.index >= test_data.index.min()]

    if train_features.empty or test_features.empty:
        raise ValueError("Train yoki test feature'lari bo'sh.")

    x_train = train_features[FEATURE_COLUMNS]
    y_train = train_features["target_change"]
    x_test = test_features[FEATURE_COLUMNS]
    actual_close = test_features["Close"]
    previous_close = test_features["lag_1"]

    predictions: dict[str, pd.Series] = {BASELINE_NAME: previous_close.copy()}
    metrics_rows: list[dict[str, float | str]] = [
        {"Model": BASELINE_NAME, **calculate_metrics(actual_close, previous_close)}
    ]

    for model_name, model_builder in get_sklearn_model_builders().items():
        model = model_builder()
        model.fit(x_train, y_train)
        predicted_change = model.predict(x_test)
        predicted_close = pd.Series(
            previous_close.to_numpy() + predicted_change,
            index=x_test.index,
        )
        predictions[model_name] = predicted_close
        metrics_rows.append(
            {"Model": model_name, **calculate_metrics(actual_close, predicted_close)}
        )

    model_prediction_frame = pd.concat(
        [predictions[model_name] for model_name in MODEL_NAMES],
        axis=1,
    )
    predictions[ENSEMBLE_NAME] = model_prediction_frame.mean(axis=1)
    predictions[BLEND_NAME] = (predictions[ENSEMBLE_NAME] + previous_close) / 2
    metrics_rows.extend(
        [
            {
                "Model": ENSEMBLE_NAME,
                **calculate_metrics(actual_close, predictions[ENSEMBLE_NAME]),
            },
            {
                "Model": BLEND_NAME,
                **calculate_metrics(actual_close, predictions[BLEND_NAME]),
            },
        ]
    )

    metrics_table = pd.DataFrame(metrics_rows).sort_values(
        by=["RMSE", "MAE"],
        ascending=True,
    )
    best_model_name = str(metrics_table.iloc[0]["Model"])
    return metrics_table.reset_index(drop=True), predictions, best_model_name


def evaluate_models_with_walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    """Vaqt tartibini saqlagan holda 4 fold walk-forward natijalarini hisoblaydi."""
    feature_frame = build_feature_frame(data)
    x_all = feature_frame[FEATURE_COLUMNS]
    y_all = feature_frame["target_change"]
    actual_close = feature_frame["Close"]
    previous_close = feature_frame["lag_1"]
    splitter = TimeSeriesSplit(n_splits=4)
    fold_rows: list[dict[str, float | str]] = []

    for fold_number, (train_index, test_index) in enumerate(splitter.split(x_all), start=1):
        x_train = x_all.iloc[train_index]
        y_train = y_all.iloc[train_index]
        x_test = x_all.iloc[test_index]
        actual_fold = actual_close.iloc[test_index]
        previous_fold = previous_close.iloc[test_index]

        baseline_metrics = calculate_metrics(actual_fold, previous_fold)
        fold_rows.append(
            {
                "Model": BASELINE_NAME,
                "Fold": fold_number,
                "RMSE": baseline_metrics["RMSE"],
                "MAPE": baseline_metrics["MAPE"],
            }
        )

        for model_name, model_builder in get_sklearn_model_builders().items():
            model = model_builder()
            model.fit(x_train, y_train)
            predicted_change = model.predict(x_test)
            predicted_close = pd.Series(
                previous_fold.to_numpy() + predicted_change,
                index=x_test.index,
            )
            metrics = calculate_metrics(actual_fold, predicted_close)
            fold_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold_number,
                    "RMSE": metrics["RMSE"],
                    "MAPE": metrics["MAPE"],
                }
            )

    fold_table = pd.DataFrame(fold_rows)
    return (
        fold_table.groupby("Model", as_index=False)
        .agg(
            WalkForward_RMSE=("RMSE", "mean"),
            WalkForward_MAPE=("MAPE", "mean"),
        )
        .sort_values("WalkForward_RMSE", ascending=True)
        .reset_index(drop=True)
    )


def build_weekly_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Uzoq muddat forecast uchun haftalik close va return feature'larini yaratadi."""
    weekly = data["Close"].resample("W-FRI").last().dropna().to_frame("Close")
    weekly["return_1w"] = weekly["Close"].pct_change()

    for lag in range(1, WEEKLY_RETURN_LAGS + 1):
        weekly[f"return_lag_{lag}"] = weekly["return_1w"].shift(lag)

    for window in WEEKLY_WINDOWS:
        weekly[f"return_mean_{window}"] = (
            weekly["return_1w"].shift(1).rolling(window).mean()
        )
        weekly[f"return_std_{window}"] = (
            weekly["return_1w"].shift(1).rolling(window).std()
        )
        weekly[f"price_momentum_{window}"] = (
            weekly["Close"].shift(1) / weekly["Close"].shift(window + 1) - 1
        )

    weekly["week_index"] = np.arange(len(weekly))
    return weekly.dropna()


def get_weekly_feature_columns() -> list[str]:
    """Haftalik forecast modelida ishlatiladigan feature nomlari."""
    return [
        *[f"return_lag_{lag}" for lag in range(1, WEEKLY_RETURN_LAGS + 1)],
        *[f"return_mean_{window}" for window in WEEKLY_WINDOWS],
        *[f"return_std_{window}" for window in WEEKLY_WINDOWS],
        *[f"price_momentum_{window}" for window in WEEKLY_WINDOWS],
        "week_index",
    ]


def build_weekly_target_matrix(
    weekly_frame: pd.DataFrame,
    horizon_weeks: int,
) -> pd.DataFrame:
    """Keyingi haftalarning returnlarini multi-output target sifatida yaratadi."""
    targets = {
        f"target_return_{step}": weekly_frame["return_1w"].shift(-step)
        for step in range(1, horizon_weeks + 1)
    }
    return pd.DataFrame(targets, index=weekly_frame.index)


def build_weekly_sklearn_model(model_name: str) -> object:
    """Haftalik multi-output forecast uchun klassik ML modelini yaratadi."""
    if model_name == "Ridge Regression":
        return Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=2.0))]
        )
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        )
    if model_name == "Gradient Boosting Regressor":
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=90,
                learning_rate=0.03,
                max_depth=2,
                random_state=42,
            )
        )
    if model_name == "Extra Trees Regressor":
        return ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        )
    raise KeyError(model_name)


def returns_to_price_path(
    start_price: float,
    predicted_returns: np.ndarray,
) -> np.ndarray:
    """Haftalik returnlar ketma-ketligini narx yo'liga aylantiradi."""
    clipped_returns = np.clip(predicted_returns.astype(float), -0.25, 0.25)
    return start_price * np.cumprod(1 + clipped_returns)


def forecast_weekly_sklearn(
    data: pd.DataFrame,
    model_name: str,
    horizon_weeks: int,
) -> pd.Series:
    """Klassik ML modeli bilan haftalik direct multi-output forecast."""
    weekly_frame = build_weekly_frame(data)
    feature_columns = get_weekly_feature_columns()
    target_matrix = build_weekly_target_matrix(weekly_frame, horizon_weeks)
    train_frame = weekly_frame.join(target_matrix).dropna()
    model = build_weekly_sklearn_model(model_name)
    model.fit(train_frame[feature_columns], train_frame[target_matrix.columns])
    latest_features = weekly_frame.iloc[[-1]][feature_columns]
    predicted_returns = np.asarray(model.predict(latest_features)).reshape(-1)
    predicted_prices = returns_to_price_path(
        float(weekly_frame["Close"].iloc[-1]),
        predicted_returns,
    )
    future_dates = pd.date_range(
        start=weekly_frame.index.max() + pd.offsets.Week(weekday=4),
        periods=horizon_weeks,
        freq="W-FRI",
    )
    return pd.Series(predicted_prices, index=future_dates)


def get_horizon_weeks(forecast_months: int) -> int:
    """Oy sonini taxminiy trading haftalariga aylantiradi."""
    return {3: 13, 6: 26, 12: 52}[forecast_months]


def forecast_weekly_baseline(data: pd.DataFrame, horizon_weeks: int) -> pd.Series:
    """Kelajak uchun flat-price benchmark prognozini yaratadi."""
    weekly_close = data["Close"].resample("W-FRI").last().dropna()
    future_dates = pd.date_range(
        start=weekly_close.index.max() + pd.offsets.Week(weekday=4),
        periods=horizon_weeks,
        freq="W-FRI",
    )
    return pd.Series(float(weekly_close.iloc[-1]), index=future_dates)


def build_weekly_forecast_bundle(
    data: pd.DataFrame,
    horizon_weeks: int,
) -> pd.DataFrame:
    """Benchmark va 4 ta model prognozlarini bitta jadvalga yig'adi."""
    forecasts = {
        BASELINE_NAME: forecast_weekly_baseline(data, horizon_weeks),
        **{
            model_name: forecast_weekly_sklearn(data, model_name, horizon_weeks)
            for model_name in MODEL_NAMES
        },
    }
    return add_derived_forecast_columns(pd.DataFrame(forecasts))


def add_derived_forecast_columns(forecast_frame: pd.DataFrame) -> pd.DataFrame:
    """Eski cache natijalariga ham ansambl ustunlarini qo'shadi."""
    updated_frame = forecast_frame.copy()
    if ENSEMBLE_NAME not in updated_frame:
        updated_frame[ENSEMBLE_NAME] = updated_frame[MODEL_NAMES].mean(axis=1)
    if BLEND_NAME not in updated_frame:
        updated_frame[BLEND_NAME] = (
            updated_frame[ENSEMBLE_NAME] + updated_frame[BASELINE_NAME]
        ) / 2
    return updated_frame


def add_derived_prediction_series(
    predictions: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Eski cache natijalariga ham kunlik ansambl seriyalarini qo'shadi."""
    updated_predictions = predictions.copy()
    if ENSEMBLE_NAME not in updated_predictions:
        updated_predictions[ENSEMBLE_NAME] = pd.concat(
            [updated_predictions[model_name] for model_name in MODEL_NAMES],
            axis=1,
        ).mean(axis=1)
    if BLEND_NAME not in updated_predictions:
        updated_predictions[BLEND_NAME] = (
            updated_predictions[ENSEMBLE_NAME] + updated_predictions[BASELINE_NAME]
        ) / 2
    return updated_predictions


def add_derived_metric_rows(
    metrics_table: pd.DataFrame,
    actual_close: pd.Series,
    predictions: dict[str, pd.Series],
) -> tuple[pd.DataFrame, str]:
    """Eski cache natijalarida ansambl metrikalarini tiklaydi."""
    updated_table = metrics_table.copy()
    existing_models = set(updated_table["Model"])
    missing_models = [
        model_name
        for model_name in DERIVED_FORECAST_NAMES
        if model_name not in existing_models
    ]
    if missing_models:
        extra_rows = [
            {
                "Model": model_name,
                **calculate_metrics(actual_close, predictions[model_name]),
            }
            for model_name in missing_models
        ]
        updated_table = pd.concat(
            [updated_table, pd.DataFrame(extra_rows)],
            ignore_index=True,
        )
    updated_table = updated_table.sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    return updated_table, str(updated_table.iloc[0]["Model"])


def run_horizon_matched_backtest(
    data: pd.DataFrame,
    forecast_months: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Kelajak forecast bilan bir xil haftalik ufqda tarixiy backtest yaratadi."""
    horizon_weeks = get_horizon_weeks(forecast_months)
    weekly_full = data["Close"].resample("W-FRI").last().dropna()
    training_end = weekly_full.index[-(horizon_weeks + 1)]
    training_data = data.loc[:training_end].copy()
    actual_future = weekly_full.loc[weekly_full.index > training_end].iloc[:horizon_weeks]
    backtest_frame = build_weekly_forecast_bundle(training_data, horizon_weeks)
    return actual_future, backtest_frame


def evaluate_rolling_horizon_backtests(
    data: pd.DataFrame,
    forecast_months: int,
    origins: int = ROLLING_BACKTEST_ORIGINS,
) -> pd.DataFrame:
    """Bir nechta tarixiy origin bo'yicha uzoq muddat prognozlarini baholaydi."""
    horizon_weeks = get_horizon_weeks(forecast_months)
    weekly_full = data["Close"].resample("W-FRI").last().dropna()
    rows: list[dict[str, float | int | str | bool]] = []

    for origin in range(origins, 0, -1):
        offset = origin * horizon_weeks + 1
        if offset >= len(weekly_full):
            continue

        training_end = weekly_full.index[-offset]
        training_data = data.loc[:training_end].copy()
        actual_future = weekly_full.loc[weekly_full.index > training_end].iloc[:horizon_weeks]
        forecasts = build_weekly_forecast_bundle(training_data, horizon_weeks)

        if len(actual_future) != horizon_weeks or forecasts.empty:
            continue

        baseline_rmse = calculate_metrics(actual_future, forecasts[BASELINE_NAME])["RMSE"]
        for model_name in ALL_FORECAST_NAMES:
            metrics = calculate_metrics(actual_future, forecasts[model_name])
            rows.append(
                {
                    "Origin": origin,
                    "Model": model_name,
                    **metrics,
                    "Beats Benchmark": bool(metrics["RMSE"] < baseline_rmse),
                }
            )

    if not rows:
        raise ValueError("Rolling backtest uchun yetarli tarixiy ma'lumot yo'q.")

    return pd.DataFrame(rows)


def summarize_rolling_horizon_backtests(details: pd.DataFrame) -> pd.DataFrame:
    """Rolling horizon testlarini model darajasida jamlaydi."""
    baseline_rmse = float(
        details.loc[details["Model"] == BASELINE_NAME, "RMSE"].mean()
    )
    win_counts = (
        details.sort_values(["Origin", "RMSE", "MAE"])
        .groupby("Origin", as_index=False)
        .head(1)["Model"]
        .value_counts()
    )
    summary = (
        details.groupby("Model", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE=("MAPE", "mean"),
            **{"R2 Score": ("R2 Score", "mean")},
            Origins=("Origin", "nunique"),
            **{"Benchmark Wins": ("Beats Benchmark", "sum")},
        )
        .sort_values(["RMSE", "MAE"], ascending=True)
        .reset_index(drop=True)
    )
    summary["Wins"] = summary["Model"].map(win_counts).fillna(0).astype(int)
    summary["Skill vs Benchmark"] = (
        (baseline_rmse - summary["RMSE"]) / baseline_rmse * 100
    )
    return summary

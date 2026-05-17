from __future__ import annotations

import io
import logging
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable

# Ba'zi Windows tizimlarida joblib fizik yadrolarni aniqlay olmaydi.
# Shu sababli ogohlantirish chiqmasligi uchun qiymatni oldindan belgilab qo'yamiz.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
    tf.get_logger().setLevel("ERROR")
except ImportError:
    tf = None
    keras = None
    TENSORFLOW_AVAILABLE = False


# Ilovadagi asosiy sozlamalar
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL"]
MODEL_NAMES = [
    "Linear Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "LSTM Neural Network",
]
HISTORY_PERIOD = "10y"
DEMO_PERIODS = 2520
TEST_YEARS = 1
LAG_DAYS = 30
ROLLING_WINDOWS = (5, 10, 20, 30)
RETURN_PERIODS = (1, 5, 10, 20)
MOMENTUM_PERIODS = (5, 10, 20)
EMA_SPANS = (5, 10, 20, 30)
LSTM_LOOKBACK = 60
WEEKLY_LOOKBACK = 52
WEEKLY_RETURN_LAGS = 26
WEEKLY_WINDOWS = (4, 8, 13, 26)
FEATURE_COLUMNS = [
    *[f"lag_{lag}" for lag in range(1, LAG_DAYS + 1)],
    *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
    *[f"rolling_std_{window}" for window in ROLLING_WINDOWS],
    *[f"return_{period}" for period in RETURN_PERIODS],
    *[f"momentum_{period}" for period in MOMENTUM_PERIODS],
    *[f"ema_{span}" for span in EMA_SPANS],
    "day_index",
]


# Terminalni ortiqcha xabarlardan toza tutamiz.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning)


def create_demo_data(ticker: str, periods: int = DEMO_PERIODS) -> pd.DataFrame:
    """Internet bo'lmaganda ishlatish uchun 10 yillik sintetik data yaratadi."""
    end_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_date, periods=periods)

    seed = sum(ord(symbol) for symbol in ticker.upper()) + 42
    rng = np.random.default_rng(seed)
    base_price = 70 + (seed % 90)
    trend = np.linspace(0, 120, periods)
    seasonality = 10 * np.sin(np.linspace(0, 18 * np.pi, periods))
    long_noise = rng.normal(0, 1.4, periods).cumsum() * 0.6
    close_prices = np.maximum(base_price + trend + seasonality + long_noise, 5)
    open_prices = np.r_[close_prices[0], close_prices[:-1]] + rng.normal(0, 0.8, periods)
    high_prices = np.maximum(open_prices, close_prices) + rng.uniform(0.2, 2.4, periods)
    low_prices = np.minimum(open_prices, close_prices) - rng.uniform(0.2, 2.4, periods)
    volume = rng.integers(12_000_000, 85_000_000, periods)

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": np.maximum(low_prices, 0.01),
            "Close": close_prices,
            "Volume": volume,
        },
        index=dates,
    )


def normalize_stock_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """yfinance qaytargan ma'lumotdan bitta `Close` ustunini ajratadi."""
    if raw_data.empty:
        raise ValueError("Bo'sh ma'lumot qaytdi.")

    price_columns = ["Open", "High", "Low", "Close", "Volume"]
    extracted_columns: dict[str, pd.Series] = {}

    for column in price_columns:
        if isinstance(raw_data.columns, pd.MultiIndex):
            if column not in raw_data.columns.get_level_values(0):
                continue
            column_data = raw_data[column]
            extracted_columns[column] = (
                column_data.iloc[:, 0] if isinstance(column_data, pd.DataFrame) else column_data
            )
        elif column in raw_data.columns:
            extracted_columns[column] = raw_data[column]

    if "Close" not in extracted_columns:
        raise ValueError("`Close` ustuni topilmadi.")

    data = pd.DataFrame(extracted_columns).dropna(subset=["Close"])
    data.index = pd.to_datetime(data.index)
    return data.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker: str) -> tuple[pd.DataFrame, bool]:
    """Oxirgi 10 yillik data yuklaydi, bo'lmasa demo data qaytaradi."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            raw_data = yf.download(
                tickers=ticker,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=10,
            )

        data = normalize_stock_data(raw_data)
        if len(data) < 500:
            raise ValueError("Prognoz uchun yetarli ma'lumot yo'q.")
        return data, False
    except Exception:
        return create_demo_data(ticker), True


@st.cache_data(ttl=300, show_spinner=False)
def load_live_quote(ticker: str) -> dict[str, float | str | bool]:
    """Yahoo Finance'dan joriy kotirovkani oladi."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            fast_info = dict(yf.Ticker(ticker).fast_info)

        last_price = float(fast_info["lastPrice"])
        previous_close = float(
            fast_info.get("previousClose", fast_info.get("regularMarketPreviousClose"))
        )
        return {
            "last_price": last_price,
            "previous_close": previous_close,
            "day_change": last_price - previous_close,
            "day_change_pct": ((last_price - previous_close) / previous_close) * 100,
            "currency": str(fast_info.get("currency", "USD")),
            "source": "Yahoo Finance live quote",
            "is_live": True,
        }
    except Exception:
        return {
            "last_price": np.nan,
            "previous_close": np.nan,
            "day_change": np.nan,
            "day_change_pct": np.nan,
            "currency": "USD",
            "source": "Historical close fallback",
            "is_live": False,
        }


def split_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Birinchi 9 yil train, oxirgi 1 yil test sifatida ajratiladi."""
    cutoff_date = data.index.max() - pd.DateOffset(years=TEST_YEARS)
    train_data = data[data.index < cutoff_date].copy()
    test_data = data[data.index >= cutoff_date].copy()

    if len(train_data) <= LAG_DAYS + 30 or len(test_data) < 30:
        split_index = int(len(data) * 0.9)
        train_data = data.iloc[:split_index].copy()
        test_data = data.iloc[split_index:].copy()

    return train_data, test_data


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """ML/DL modellar uchun lag, trend va volatilitet feature'larini yaratadi."""
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
    """Ilovadagi klassik ML modellarini yaratadi."""
    return {
        "Linear Regression": lambda: Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Random Forest Regressor": lambda: RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
        "Gradient Boosting Regressor": lambda: GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            random_state=42,
        ),
    }


def build_lstm_model(output_units: int) -> object:
    """Bir qavatli yengil LSTM modelini yaratadi."""
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow o'rnatilmagan.")

    tf.keras.utils.set_random_seed(42)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(LSTM_LOOKBACK, 1)),
            keras.layers.LSTM(48),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(output_units),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def create_lstm_one_step_dataset(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index, MinMaxScaler]:
    """LSTM uchun one-step train/test sequence to'plamlarini yaratadi."""
    scaler = MinMaxScaler()
    scaler.fit(train_data[["Close"]])
    scaled_close = scaler.transform(data[["Close"]]).astype(np.float32).flatten()
    train_last_position = data.index.get_loc(train_data.index.max())
    test_first_position = data.index.get_loc(test_data.index.min())

    x_train, y_train = [], []
    for target_position in range(LSTM_LOOKBACK, train_last_position + 1):
        x_train.append(scaled_close[target_position - LSTM_LOOKBACK : target_position])
        y_train.append(scaled_close[target_position])

    x_test = []
    test_index = data.index[test_first_position:]
    for target_position in range(test_first_position, len(data)):
        x_test.append(scaled_close[target_position - LSTM_LOOKBACK : target_position])

    return (
        np.asarray(x_train, dtype=np.float32).reshape(-1, LSTM_LOOKBACK, 1),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(x_test, dtype=np.float32).reshape(-1, LSTM_LOOKBACK, 1),
        test_index,
        scaler,
    )


def predict_lstm_holdout(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.Series:
    """Oxirgi 1 yil uchun LSTM one-step bashoratini hisoblaydi."""
    x_train, y_train, x_test, test_index, scaler = create_lstm_one_step_dataset(
        data,
        train_data,
        test_data,
    )
    model = build_lstm_model(output_units=1)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        x_train,
        y_train,
        epochs=25,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )
    predictions = model.predict(x_test, verbose=0).reshape(-1, 1)
    close_predictions = scaler.inverse_transform(predictions).flatten()
    return pd.Series(close_predictions, index=test_index)


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Model sifatini MAE, RMSE, MAPE va R² orqali hisoblaydi."""
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    safe_actual = np.where(actual_values == 0, np.nan, actual_values)

    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "MAPE": float(
            np.nanmean(np.abs((actual_values - predicted_values) / safe_actual)) * 100
        ),
        "R² Score": float(r2_score(actual_values, predicted_values)),
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

    predictions: dict[str, pd.Series] = {}
    metrics_rows: list[dict[str, float | str]] = []

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

    lstm_prediction = predict_lstm_holdout(data, train_data, test_data)
    lstm_actual = data.loc[lstm_prediction.index, "Close"]
    predictions["LSTM Neural Network"] = lstm_prediction
    metrics_rows.append(
        {"Model": "LSTM Neural Network", **calculate_metrics(lstm_actual, lstm_prediction)}
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
    summary = (
        fold_table.groupby("Model", as_index=False)
        .agg(
            WalkForward_RMSE=("RMSE", "mean"),
            WalkForward_MAPE=("MAPE", "mean"),
        )
        .sort_values("WalkForward_RMSE", ascending=True)
        .reset_index(drop=True)
    )
    return summary


@st.cache_data(show_spinner=False)
def run_model_evaluation(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Series], str]:
    """Sidebar o'zgarishlarida modellarni qayta-qayta o'qitmaslik uchun cache wrapper."""
    train_data, test_data = split_train_test(data)
    metrics_table, predictions, best_model_name = evaluate_models(
        data=data,
        train_data=train_data,
        test_data=test_data,
    )
    return train_data, test_data, metrics_table, predictions, best_model_name


@st.cache_data(show_spinner=False)
def run_walk_forward_validation(data: pd.DataFrame) -> pd.DataFrame:
    """Klassik ML modellar uchun qo'shimcha walk-forward tekshiruv."""
    return evaluate_models_with_walk_forward(data)


def build_future_feature_row(history: list[float]) -> dict[str, float]:
    """Kelajakdagi bitta kun uchun feature'lar hosil qiladi."""
    history_series = pd.Series(history, dtype=float)
    feature_row: dict[str, float] = {}

    for lag in range(1, LAG_DAYS + 1):
        feature_row[f"lag_{lag}"] = float(history[-lag])

    for window in ROLLING_WINDOWS:
        recent_values = history[-window:]
        feature_row[f"rolling_mean_{window}"] = float(np.mean(recent_values))
        feature_row[f"rolling_std_{window}"] = float(np.std(recent_values, ddof=1))

    for period in RETURN_PERIODS:
        feature_row[f"return_{period}"] = float(
            history_series.pct_change(period).iloc[-1]
        )

    for period in MOMENTUM_PERIODS:
        feature_row[f"momentum_{period}"] = float(history[-1] - history[-(period + 1)])

    for span in EMA_SPANS:
        feature_row[f"ema_{span}"] = float(
            history_series.ewm(span=span, adjust=False).mean().iloc[-1]
        )

    feature_row["day_index"] = float(len(history))
    return feature_row


def fit_model_on_all_data(model_name: str, data: pd.DataFrame) -> object:
    """Kelajak prognozi uchun tanlangan modelni barcha 10 yillik data bilan o'qitadi."""
    feature_frame = build_feature_frame(data)
    model = get_sklearn_model_builders()[model_name]()
    model.fit(feature_frame[FEATURE_COLUMNS], feature_frame["target_change"])
    return model


def get_daily_change_limit(data: pd.DataFrame) -> float:
    """Juda keskin va real bo'lmagan sakrashlarni cheklash uchun limit beradi."""
    historical_changes = data["Close"].diff().abs().dropna()
    return float(max(historical_changes.quantile(0.99), 1.0))


def forecast_future(
    data: pd.DataFrame,
    model_name: str,
    forecast_months: int,
) -> pd.DataFrame:
    """Tanlangan model bilan 3/6/12 oylik rekursiv prognoz yaratadi."""
    last_date = data.index.max()
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        end=last_date + pd.DateOffset(months=forecast_months),
    )
    fitted_model = fit_model_on_all_data(model_name, data)
    history = data["Close"].astype(float).tolist()
    future_values: list[float] = []
    daily_change_limit = get_daily_change_limit(data)

    for _future_date in future_dates:
        feature_row = build_future_feature_row(history)
        feature_df = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
        predicted_change = float(fitted_model.predict(feature_df)[0])
        predicted_change = float(
            np.clip(predicted_change, -daily_change_limit, daily_change_limit)
        )
        next_value = max(history[-1] + predicted_change, 0.01)

        history.append(next_value)
        future_values.append(next_value)

    return pd.DataFrame({"Forecast": future_values}, index=future_dates)


def forecast_future_direct_sklearn(
    data: pd.DataFrame,
    model_name: str,
    forecast_months: int,
) -> pd.DataFrame:
    """Uzoq muddat uchun direct multi-horizon prognoz yaratadi."""
    feature_frame = build_feature_frame(data)
    latest_features = feature_frame.iloc[-1][FEATURE_COLUMNS]
    latest_date = data.index.max()
    latest_close = float(data["Close"].iloc[-1])
    horizon_days = [21 * month for month in range(1, forecast_months + 1)]
    anchor_dates = [latest_date + pd.offsets.BDay(days) for days in horizon_days]
    anchor_values: list[float] = []

    for horizon in horizon_days:
        horizon_frame = feature_frame.copy()
        horizon_frame["target_close_horizon"] = data["Close"].shift(-horizon).reindex(
            horizon_frame.index
        )
        horizon_frame = horizon_frame.dropna()

        model = get_sklearn_model_builders()[model_name]()
        model.fit(
            horizon_frame[FEATURE_COLUMNS],
            horizon_frame["target_close_horizon"],
        )
        prediction = float(
            model.predict(pd.DataFrame([latest_features], columns=FEATURE_COLUMNS))[0]
        )
        anchor_values.append(max(prediction, 0.01))

    future_dates = pd.bdate_range(
        start=latest_date + pd.offsets.BDay(1),
        end=anchor_dates[-1],
    )
    anchor_series = pd.Series(
        [latest_close, *anchor_values],
        index=[latest_date, *anchor_dates],
        dtype=float,
    )
    forecast_series = (
        anchor_series.reindex(anchor_series.index.union(future_dates))
        .sort_index()
        .interpolate(method="time")
        .reindex(future_dates)
    )
    return pd.DataFrame({"Forecast": forecast_series}, index=future_dates)


def create_lstm_multi_horizon_dataset(
    data: pd.DataFrame,
    forecast_months: int,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Kelajakdagi oylik anchor nuqtalar uchun LSTM train data yaratadi."""
    horizon_days = [21 * month for month in range(1, forecast_months + 1)]
    max_horizon = max(horizon_days)
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(data[["Close"]]).astype(np.float32).flatten()

    x_train, y_train = [], []
    for end_position in range(LSTM_LOOKBACK, len(scaled_close) - max_horizon + 1):
        x_train.append(scaled_close[end_position - LSTM_LOOKBACK : end_position])
        y_train.append(
            [scaled_close[end_position + horizon - 1] for horizon in horizon_days]
        )

    return (
        np.asarray(x_train, dtype=np.float32).reshape(-1, LSTM_LOOKBACK, 1),
        np.asarray(y_train, dtype=np.float32),
        scaler,
    )


def forecast_future_direct_lstm(
    data: pd.DataFrame,
    forecast_months: int,
) -> pd.DataFrame:
    """LSTM yordamida direct multi-horizon kelajak prognozi yaratadi."""
    x_train, y_train, scaler = create_lstm_multi_horizon_dataset(data, forecast_months)
    model = build_lstm_model(output_units=forecast_months)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        x_train,
        y_train,
        epochs=25,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )

    latest_window = scaler.transform(data[["Close"]].tail(LSTM_LOOKBACK)).astype(
        np.float32
    )
    scaled_anchors = model.predict(
        latest_window.reshape(1, LSTM_LOOKBACK, 1),
        verbose=0,
    ).reshape(-1, 1)
    anchor_values = scaler.inverse_transform(scaled_anchors).flatten()

    latest_date = data.index.max()
    anchor_dates = [
        latest_date + pd.offsets.BDay(21 * month)
        for month in range(1, forecast_months + 1)
    ]
    future_dates = pd.bdate_range(
        start=latest_date + pd.offsets.BDay(1),
        end=anchor_dates[-1],
    )
    anchor_series = pd.Series(
        [float(data["Close"].iloc[-1]), *anchor_values],
        index=[latest_date, *anchor_dates],
        dtype=float,
    )
    forecast_series = (
        anchor_series.reindex(anchor_series.index.union(future_dates))
        .sort_index()
        .interpolate(method="time")
        .reindex(future_dates)
    )
    return pd.DataFrame({"Forecast": forecast_series}, index=future_dates)


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
    if model_name == "Linear Regression":
        return Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        )
    if model_name == "Gradient Boosting Regressor":
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=2,
                random_state=42,
            )
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


def create_weekly_lstm_dataset(
    weekly_frame: pd.DataFrame,
    horizon_weeks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Haftalik returnlar asosida LSTM multi-output dataset yaratadi."""
    return_values = weekly_frame["return_1w"].to_numpy(dtype=np.float32)
    x_train, y_train = [], []
    for end_position in range(
        WEEKLY_LOOKBACK,
        len(return_values) - horizon_weeks + 1,
    ):
        x_train.append(return_values[end_position - WEEKLY_LOOKBACK : end_position])
        y_train.append(return_values[end_position : end_position + horizon_weeks])
    return (
        np.asarray(x_train, dtype=np.float32).reshape(-1, WEEKLY_LOOKBACK, 1),
        np.asarray(y_train, dtype=np.float32),
    )


def build_weekly_lstm_model(output_units: int) -> object:
    """Haftalik direct multi-output forecast uchun LSTM modeli."""
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow o'rnatilmagan.")
    tf.keras.utils.set_random_seed(42)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(WEEKLY_LOOKBACK, 1)),
            keras.layers.LSTM(48),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(output_units),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def forecast_weekly_lstm(
    data: pd.DataFrame,
    horizon_weeks: int,
) -> pd.Series:
    """LSTM bilan haftalik direct multi-output forecast."""
    weekly_frame = build_weekly_frame(data)
    x_train, y_train = create_weekly_lstm_dataset(weekly_frame, horizon_weeks)
    model = build_weekly_lstm_model(horizon_weeks)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        x_train,
        y_train,
        epochs=35,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )
    latest_window = weekly_frame["return_1w"].tail(WEEKLY_LOOKBACK).to_numpy(
        dtype=np.float32
    )
    predicted_returns = model.predict(
        latest_window.reshape(1, WEEKLY_LOOKBACK, 1),
        verbose=0,
    ).reshape(-1)
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


@st.cache_data(show_spinner=False)
def run_all_future_forecasts(
    data: pd.DataFrame,
    forecast_months: int,
) -> pd.DataFrame:
    """4 ta model uchun haftalik multi-horizon prognozlarni yig'adi."""
    horizon_weeks = get_horizon_weeks(forecast_months)
    forecasts = {
        model_name: forecast_weekly_sklearn(data, model_name, horizon_weeks)
        for model_name in get_sklearn_model_builders()
    }
    forecasts["LSTM Neural Network"] = forecast_weekly_lstm(
        data,
        horizon_weeks,
    )
    return pd.DataFrame(forecasts)


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
    backtest_forecasts = {
        model_name: forecast_weekly_sklearn(
            training_data,
            model_name,
            horizon_weeks,
        )
        for model_name in get_sklearn_model_builders()
    }
    backtest_forecasts["LSTM Neural Network"] = forecast_weekly_lstm(
        training_data,
        horizon_weeks,
    )
    backtest_frame = pd.DataFrame(backtest_forecasts)
    return actual_future, backtest_frame


@st.cache_data(show_spinner=False)
def run_cached_horizon_backtest(
    data: pd.DataFrame,
    forecast_months: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Horizon-matched backtestni cache qiladi."""
    return run_horizon_matched_backtest(data, forecast_months)


def apply_chart_style(figure: go.Figure, title: str) -> go.Figure:
    """Barcha grafiklar uchun bir xil zamonaviy uslub beradi."""
    figure.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.96)",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(148,163,184,0.24)")
    return figure


def create_price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Tanlangan aksiyaning real candlestick chartini ko'rsatadi."""
    chart_data = data.tail(252).copy()
    chart_data["ema_20"] = chart_data["Close"].ewm(span=20, adjust=False).mean()
    chart_data["ema_50"] = chart_data["Close"].ewm(span=50, adjust=False).mean()

    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="OHLC",
            increasing_line_color="#16A34A",
            decreasing_line_color="#DC2626",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["ema_20"],
            mode="lines",
            name="EMA 20",
            line=dict(color="#F97316", width=1.8, dash="dot"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["ema_50"],
            mode="lines",
            name="EMA 50",
            line=dict(color="#7C3AED", width=1.8, dash="dot"),
        )
    )
    styled = apply_chart_style(figure, f"{ticker} — real candlestick chart (oxirgi 12 oy)")
    styled.update_layout(xaxis_rangeslider_visible=False)
    return styled


def create_train_test_chart(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> go.Figure:
    """Train va test davrlarini alohida ranglarda ko'rsatadi."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=train_data.index,
            y=train_data["Close"],
            mode="lines",
            name="Train data",
            line=dict(color="#0F766E", width=2.2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data["Close"],
            mode="lines",
            name="Test data",
            line=dict(color="#F97316", width=2.4),
        )
    )
    return apply_chart_style(figure, "Train va test davrlari")


def create_prediction_chart(
    actual: pd.Series,
    predicted: pd.Series,
    model_name: str,
) -> go.Figure:
    """Test davridagi haqiqiy narx va model bashoratini taqqoslaydi."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual,
            mode="lines",
            name="Haqiqiy narx",
            line=dict(color="#111827", width=2.4),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=predicted.index,
            y=predicted,
            mode="lines",
            name="Model bashorati",
            line=dict(color="#DC2626", width=2.2, dash="dash"),
        )
    )
    return apply_chart_style(
        figure,
        f"Oxirgi 1 yil: haqiqiy narx va {model_name} bashorati",
    )


def create_model_comparison_chart(metrics_table: pd.DataFrame) -> go.Figure:
    """RMSE bo'yicha modellar reytingini beradi."""
    chart_data = metrics_table.sort_values("RMSE", ascending=True)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=chart_data["RMSE"],
            y=chart_data["Model"],
            orientation="h",
            marker=dict(
                color=["#7C3AED", "#2563EB", "#0F766E", "#F97316"],
                line=dict(width=0),
            ),
            text=chart_data["RMSE"].map(lambda value: f"{value:.2f}"),
            textposition="outside",
        )
    )
    figure.update_yaxes(autorange="reversed")
    return apply_chart_style(figure, "RMSE bo'yicha modellar reytingi")


def create_backtest_chart(
    actual: pd.Series,
    predictions: dict[str, pd.Series],
) -> go.Figure:
    """Oxirgi 1 yillik test davrida barcha modellarni bitta grafikda ko'rsatadi."""
    colors = {
        "Linear Regression": "#2563EB",
        "Random Forest Regressor": "#0F766E",
        "Gradient Boosting Regressor": "#F97316",
        "LSTM Neural Network": "#7C3AED",
    }
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual,
            mode="lines",
            name="Haqiqiy narx",
            line=dict(color="#0F172A", width=3),
        )
    )
    for model_name in MODEL_NAMES:
        figure.add_trace(
            go.Scatter(
                x=predictions[model_name].index,
                y=predictions[model_name],
                mode="lines",
                name=model_name,
                line=dict(color=colors[model_name], width=1.9, dash="dash"),
            )
        )
    return apply_chart_style(figure, "Backtest: haqiqiy narx va 4 model")


def create_horizon_backtest_chart(
    actual: pd.Series,
    forecasts: pd.DataFrame,
) -> go.Figure:
    """Kelajak forecast bilan bir xil ufqdagi tarixiy backtest charti."""
    colors = {
        "Linear Regression": "#2563EB",
        "Random Forest Regressor": "#0F766E",
        "Gradient Boosting Regressor": "#F97316",
        "LSTM Neural Network": "#7C3AED",
    }
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual,
            mode="lines",
            name="Haqiqiy narx",
            line=dict(color="#0F172A", width=3),
        )
    )
    for model_name in MODEL_NAMES:
        figure.add_trace(
            go.Scatter(
                x=forecasts.index,
                y=forecasts[model_name],
                mode="lines",
                name=model_name,
                line=dict(color=colors[model_name], width=2.1, dash="dash"),
            )
        )
    return apply_chart_style(
        figure,
        "Horizon-matched backtest: future usuli tarixda qanday ishlagan?",
    )


def create_horizon_metrics_table(
    actual: pd.Series,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Future usuliga mos backtest metrikalarini hisoblaydi."""
    rows = []
    for model_name in MODEL_NAMES:
        metrics = calculate_metrics(actual, forecasts[model_name])
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)


def create_multi_model_forecast_chart(
    data: pd.DataFrame,
    future_forecasts: pd.DataFrame,
    best_model_name: str,
) -> go.Figure:
    """Barcha 4 modelning kelajak prognozini bitta chartda ko'rsatadi."""
    recent_history = data.tail(180)
    colors = {
        "Linear Regression": "#2563EB",
        "Random Forest Regressor": "#0F766E",
        "Gradient Boosting Regressor": "#F97316",
        "LSTM Neural Network": "#7C3AED",
    }
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recent_history.index,
            y=recent_history["Close"],
            mode="lines",
            name="Tarixiy narx",
            line=dict(color="#2563EB", width=2.4),
        )
    )
    for model_name in MODEL_NAMES:
        is_best = model_name == best_model_name
        figure.add_trace(
            go.Scatter(
                x=future_forecasts.index,
                y=future_forecasts[model_name],
                mode="lines",
                name=f"{model_name}{' ★' if is_best else ''}",
                line=dict(
                    color=colors[model_name],
                    width=3.2 if is_best else 2.2,
                    dash="solid" if is_best else "dash",
                ),
                marker=dict(size=5),
            )
        )
    return apply_chart_style(figure, "Kelajak prognozi — 4 model taqqoslanishi")


def inject_custom_css() -> None:
    """Streamlit interfeysiga yengil, zamonaviy dizayn beradi."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #0F172A;
            --muted: #475569;
            --card: rgba(255, 255, 255, 0.86);
            --border: rgba(148, 163, 184, 0.22);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 26%),
                linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.96);
        }
        [data-testid="stSidebar"] * {
            color: #F8FAFC;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
        }
        .hero {
            padding: 1.55rem 1.7rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #0F172A 0%, #312E81 55%, #2563EB 100%);
            color: white;
            box-shadow: 0 18px 48px rgba(15, 23, 42, 0.22);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            line-height: 1.05;
        }
        .hero p {
            margin: 0;
            color: rgba(255,255,255,0.82);
        }
        .hero-chip {
            display: inline-block;
            margin-top: 0.8rem;
            margin-right: 0.45rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            font-size: 0.82rem;
        }
        .kpi-card {
            padding: 1rem 1.05rem;
            border-radius: 22px;
            background: var(--card);
            border: 1px solid var(--border);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            min-height: 96px;
        }
        .kpi-label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .kpi-value {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 700;
        }
        .kpi-note {
            color: #64748B;
            font-size: 0.78rem;
            margin-top: 0.18rem;
        }
        .section-card {
            padding: 1rem;
            border-radius: 24px;
            background: var(--card);
            border: 1px solid var(--border);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }
        .control-panel {
            padding: 1rem 1.1rem;
            border-radius: 24px;
            background: rgba(255,255,255,0.82);
            border: 1px solid var(--border);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .section-title {
            color: var(--ink);
            font-size: 1.2rem;
            font-weight: 700;
            margin: 1.2rem 0 0.35rem 0;
        }
        .section-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.55rem;
        }
        .insight-card {
            padding: 1rem 1.1rem;
            border-radius: 22px;
            background: rgba(255,255,255,0.8);
            border: 1px solid var(--border);
            box-shadow: 0 10px 24px rgba(15,23,42,0.06);
            height: 100%;
        }
        .insight-title {
            color: var(--ink);
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .insight-body {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(ticker: str) -> None:
    """Sahifa tepasidagi dizayn blokini chiqaradi."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>📈 Aksiya Forecast Lab</h1>
            <p>{ticker} uchun real narx, real chart va 10 yillik data asosidagi 4 model prognozi.</p>
            <span class="hero-chip">10 yil data</span>
            <span class="hero-chip">ML + LSTM</span>
            <span class="hero-chip">3 / 6 / 12 oy forecast</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, note: str) -> None:
    """Oddiy KPI kartasini chiqaradi."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_card(title: str, body: str) -> None:
    """Metodologiyani sodda tilda tushuntiruvchi kartani chiqaradi."""
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit interfeysini ishga tushiradi."""
    st.set_page_config(
        page_title="Aksiya Forecast Lab",
        page_icon="📈",
        layout="wide",
    )
    inject_custom_css()

    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    control_columns = st.columns([1.15, 1.15, 0.9])
    with control_columns[0]:
        ticker_choice = st.selectbox(
            "Aksiya tanlang",
            options=[*DEFAULT_TICKERS, "Boshqa"],
            index=0,
        )
    with control_columns[1]:
        custom_ticker = ""
        if ticker_choice == "Boshqa":
            custom_ticker = st.text_input("Ticker kiriting", value="META")
        else:
            st.text_input("Ticker", value=ticker_choice, disabled=True)
    with control_columns[2]:
        forecast_months = st.radio(
            "Prognoz muddati",
            options=[3, 6, 12],
            index=0,
            horizontal=True,
            format_func=lambda value: f"{value} oy",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    ticker = custom_ticker.strip().upper() if ticker_choice == "Boshqa" else ticker_choice
    ticker = ticker or "AAPL"

    with st.spinner("Real narx, 10 yillik data va modellar tayyorlanmoqda..."):
        data, demo_mode = load_stock_data(ticker)
        live_quote = load_live_quote(ticker)
        train_data, test_data, metrics_table, predictions, best_model_name = (
            run_model_evaluation(data)
        )
        future_forecasts = run_all_future_forecasts(data, forecast_months)

    render_hero(ticker)

    if demo_mode:
        st.warning(
            "Demo data ishlatilmoqda: internet yoki Yahoo Finance ma'lumotlari mavjud emas."
        )

    latest_close = (
        float(live_quote["last_price"])
        if bool(live_quote["is_live"])
        else float(data["Close"].iloc[-1])
    )
    previous_close = (
        float(live_quote["previous_close"])
        if bool(live_quote["is_live"])
        else float(data["Close"].iloc[-2])
    )
    daily_delta = latest_close - previous_close
    daily_delta_pct = (daily_delta / previous_close) * 100
    horizon_actual, horizon_forecasts = run_cached_horizon_backtest(
        data,
        forecast_months,
    )
    horizon_metrics = create_horizon_metrics_table(horizon_actual, horizon_forecasts)
    best_future_model_name = str(horizon_metrics.iloc[0]["Model"])
    best_forecast_end = float(future_forecasts[best_future_model_name].iloc[-1])
    forecast_delta = best_forecast_end - latest_close

    kpi_columns = st.columns(4)
    with kpi_columns[0]:
        render_kpi_card(
            "Hozirgi real narx",
            f"${latest_close:,.2f}",
            f"{daily_delta:+.2f} ({daily_delta_pct:+.2f}%)",
        )
    with kpi_columns[1]:
        render_kpi_card(
            "Eng yaxshi uzoq muddat modeli",
            best_future_model_name,
            f"{forecast_months} oylik backtest bo'yicha",
        )
    with kpi_columns[2]:
        render_kpi_card(
            "Data hajmi",
            f"{len(data):,} kun",
            "Model treningi uchun 10 yil",
        )
    with kpi_columns[3]:
        render_kpi_card(
            f"{forecast_months} oy prognozi",
            f"${best_forecast_end:,.2f}",
            f"Joriy narxdan: {forecast_delta:+.2f}",
        )

    st.markdown('<div class="section-title">Real bozor ma’lumoti</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Tanlangan aksiyaning haqiqiy narxi va oxirgi 12 oylik real harakati.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(create_price_chart(data, ticker), use_container_width=True)

    st.markdown('<div class="section-title">Kelajak prognozi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">4 ta model bir grafikda. Bu grafik 10 yillik tarixdan o‘rganilgan haftalik multi-horizon return prognozi asosida quriladi; u faqat birinchi kun yo‘nalishini ko‘chirib ketmaydi.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        create_multi_model_forecast_chart(data, future_forecasts, best_future_model_name),
        use_container_width=True,
    )
    st.markdown('<div class="section-title">Uzoq muddat prognozi testi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Bu jadval aynan kelajak forecast usulini tarixda sinaydi. U 1 kunlik testdan ko‘ra 3/6/12 oylik prognoz sifati uchun muhimroq.</div>',
        unsafe_allow_html=True,
    )
    horizon_display = horizon_metrics.copy()
    horizon_display.insert(0, "Rank", range(1, len(horizon_display) + 1))
    for column in ["MAE", "RMSE", "MAPE", "R² Score"]:
        horizon_display[column] = horizon_display[column].map(lambda value: round(value, 4))
    st.dataframe(horizon_display, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Model sifati</div>', unsafe_allow_html=True)
    insight_columns = st.columns(3)
    with insight_columns[0]:
        render_insight_card(
            "Linear Regression",
            "Oddiy va tushunarli baseline model. Natijalarni taqqoslash uchun muhim tayanch beradi.",
        )
    with insight_columns[1]:
        render_insight_card(
            "Tree-based ML",
            "Random Forest va Gradient Boosting murakkab nolinear bog‘lanishlarni o‘rganadi.",
        )
    with insight_columns[2]:
        render_insight_card(
            "LSTM",
            "Ketma-ketlikni ko‘radigan deep learning model; vaqt qatorlari uchun aynan mos yondashuv.",
        )

    st.markdown('<div class="section-title">Metodologiya</div>', unsafe_allow_html=True)
    method_columns = st.columns(3)
    with method_columns[0]:
        render_insight_card(
            "10 yillik o‘quv data",
            "Model bir necha bozor sikllarini ko‘radi; bu qisqa tarixga qaraganda barqarorroq o‘rganishga yordam beradi.",
        )
    with method_columns[1]:
        render_insight_card(
            "Walk-forward tekshiruv",
            "Qo‘shimcha ilmiy tekshiruv sifatida vaqt tartibi saqlanadi va klassik ML modellar bir nechta tarixiy kesimlarda sinovdan o‘tadi.",
        )
    with method_columns[2]:
        render_insight_card(
            "Holdout test",
            "Oxirgi 1 yil alohida qoldiriladi; jadvaldagi asosiy natijalar aynan shu real sinov davridan olinadi.",
        )

    st.markdown('<div class="section-title">Test natijalari</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Oxirgi 1 yillik holdout davrida tekshirilgan real metrikalar.</div>',
        unsafe_allow_html=True,
    )
    display_table = metrics_table.copy()
    display_table.insert(0, "Rank", range(1, len(display_table) + 1))
    display_table["Holat"] = display_table["Model"].map(
        lambda model_name: "Eng yaxshi" if model_name == best_model_name else ""
    )
    for column in ["MAE", "RMSE", "MAPE", "R² Score"]:
        display_table[column] = display_table[column].map(lambda value: round(value, 4))
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    if best_model_name != "LSTM Neural Network":
        st.info(
            "Murakkab model har doim ham eng yaxshi natija bermaydi: aksiya narxlari juda "
            "shovqinli bo‘lgani uchun bu tickerda soddaroq model LSTMdan yaxshi ishlashi mumkin."
        )
    st.download_button(
        "Test natijalarini CSV yuklab olish",
        data=display_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_model_metrics.csv",
        mime="text/csv",
    )

    with st.expander("Qo‘shimcha model tekshiruvlarini ko‘rish"):
        left_column, right_column = st.columns(2)
        with left_column:
            st.plotly_chart(
                create_train_test_chart(train_data, test_data),
                use_container_width=True,
            )
        with right_column:
            selected_actual = test_data.loc[predictions[best_model_name].index, "Close"]
            st.plotly_chart(
                create_prediction_chart(
                    actual=selected_actual,
                    predicted=predictions[best_model_name],
                    model_name=best_model_name,
                ),
                use_container_width=True,
            )
        st.plotly_chart(
            create_backtest_chart(
                actual=test_data.loc[predictions[best_model_name].index, "Close"],
                predictions=predictions,
            ),
            use_container_width=True,
        )
        st.markdown("#### Future usuliga teng backtest")
        st.caption(
            "Bu chart kelajak forecast bilan aynan bir xil usulni tarixga qo‘llaydi. "
            "Shuning uchun prognoz qanchalik real ishlashini ko‘rsatishda 1 kunlik testdan ko‘ra halolroq."
        )
        st.plotly_chart(
            create_horizon_backtest_chart(horizon_actual, horizon_forecasts),
            use_container_width=True,
        )
        st.markdown("#### Klassik ML uchun walk-forward validation")
        st.caption(
            "Bu qo‘shimcha tekshiruv `Linear Regression`, `Random Forest` va "
            "`Gradient Boosting` modellarining vaqt bo‘yicha barqarorligini ko‘rsatadi. "
            "`LSTM` uchun asosiy taqqoslash holdout test jadvalida berilgan."
        )
        if st.button("Walk-forward tekshiruvni hisoblash"):
            walk_forward_table = run_walk_forward_validation(data)
            walk_forward_display = walk_forward_table.copy()
            walk_forward_display.insert(
                0,
                "Rank",
                range(1, len(walk_forward_display) + 1),
            )
            for column in ["WalkForward_RMSE", "WalkForward_MAPE"]:
                walk_forward_display[column] = walk_forward_display[column].map(
                    lambda value: round(value, 4)
                )
            st.dataframe(walk_forward_display, use_container_width=True, hide_index=True)

    st.download_button(
        "Prognozlarni CSV yuklab olish",
        data=future_forecasts.reset_index(names="Date").to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_{forecast_months}oy_forecast.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

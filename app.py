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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
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


# Ilovadagi asosiy sozlamalar
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL"]
MODEL_NAMES = [
    "Ridge Regression",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "Extra Trees Regressor",
]
BASELINE_NAME = "Naive Baseline"
ALL_FORECAST_NAMES = [BASELINE_NAME, *MODEL_NAMES]
HISTORY_YEARS = (2, 5, 10)
DEFAULT_HISTORY_YEARS = 5
DEMO_PERIODS_PER_YEAR = 252
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


# Terminalni ortiqcha xabarlardan toza tutamiz.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning)


def create_demo_data(ticker: str, periods: int) -> pd.DataFrame:
    """Internet bo'lmaganda ishlatish uchun sintetik data yaratadi."""
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
def load_stock_data(ticker: str, history_years: int) -> tuple[pd.DataFrame, bool]:
    """Tanlangan tarix chuqurligidagi data yuklaydi, bo'lmasa demo data qaytaradi."""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            raw_data = yf.download(
                tickers=ticker,
                period=f"{history_years}y",
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
        return create_demo_data(ticker, history_years * DEMO_PERIODS_PER_YEAR), True


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
    return pd.DataFrame(forecasts)


@st.cache_data(show_spinner=False)
def run_all_future_forecasts(
    data: pd.DataFrame,
    forecast_months: int,
) -> pd.DataFrame:
    """Benchmark va 4 ta model uchun haftalik prognozlarni yig'adi."""
    horizon_weeks = get_horizon_weeks(forecast_months)
    return build_weekly_forecast_bundle(data, horizon_weeks)


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


@st.cache_data(show_spinner=False)
def run_cached_horizon_backtest(
    data: pd.DataFrame,
    forecast_months: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Horizon-matched backtestni cache qiladi."""
    return run_horizon_matched_backtest(data, forecast_months)


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
            **{"R² Score": ("R² Score", "mean")},
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


@st.cache_data(show_spinner=False)
def run_cached_rolling_horizon_backtests(
    data: pd.DataFrame,
    forecast_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling horizon tafsiloti va jamlanmasini cache qiladi."""
    details = evaluate_rolling_horizon_backtests(data, forecast_months)
    return summarize_rolling_horizon_backtests(details), details


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
    styled.update_layout(title=f"{ticker} narxi", xaxis_rangeslider_visible=False)
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


def create_backtest_chart(
    actual: pd.Series,
    predictions: dict[str, pd.Series],
    best_model_name: str,
) -> go.Figure:
    """Oxirgi 1 yillik test davrida benchmark va modellarni ko'rsatadi."""
    colors = {
        BASELINE_NAME: "#64748B",
        "Ridge Regression": "#0F766E",
        "Random Forest Regressor": "#2563EB",
        "Gradient Boosting Regressor": "#F97316",
        "Extra Trees Regressor": "#7C3AED",
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
    for model_name in ALL_FORECAST_NAMES:
        is_best = model_name == best_model_name
        is_baseline = model_name == BASELINE_NAME
        figure.add_trace(
            go.Scatter(
                x=predictions[model_name].index,
                y=predictions[model_name],
                mode="lines",
                name=f"{model_name}{' ★' if is_best else ''}",
                line=dict(
                    color=colors[model_name],
                    width=3.0 if is_best else 2.2 if is_baseline else 1.7,
                    dash=(
                        "dot"
                        if is_baseline and not is_best
                        else "solid"
                        if is_best
                        else "dash"
                    ),
                ),
                opacity=1.0 if is_best or is_baseline else 0.7,
            )
        )
    return apply_chart_style(figure, "Kunlik backtest: haqiqiy narx, benchmark va modellar")


def create_horizon_backtest_chart(
    actual: pd.Series,
    forecasts: pd.DataFrame,
) -> go.Figure:
    """Kelajak forecast bilan bir xil ufqdagi tarixiy backtest charti."""
    colors = {
        BASELINE_NAME: "#64748B",
        "Ridge Regression": "#0F766E",
        "Random Forest Regressor": "#2563EB",
        "Gradient Boosting Regressor": "#F97316",
        "Extra Trees Regressor": "#7C3AED",
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
    for model_name in ALL_FORECAST_NAMES:
        figure.add_trace(
            go.Scatter(
                x=forecasts.index,
                y=forecasts[model_name],
                mode="lines",
                name=model_name,
                line=dict(
                    color=colors[model_name],
                    width=2.3 if model_name == BASELINE_NAME else 2.1,
                    dash="dot" if model_name == BASELINE_NAME else "dash",
                ),
            )
        )
    return apply_chart_style(
        figure,
        "Horizon-matched backtest: future usuli tarixda qanday ishlagan?",
    )


def create_multi_model_forecast_chart(
    data: pd.DataFrame,
    future_forecasts: pd.DataFrame,
    best_model_name: str,
) -> go.Figure:
    """Benchmark va 4 ta modelning kelajak prognozini bitta chartda ko'rsatadi."""
    recent_history = data.tail(180)
    colors = {
        BASELINE_NAME: "#64748B",
        "Ridge Regression": "#0F766E",
        "Random Forest Regressor": "#2563EB",
        "Gradient Boosting Regressor": "#F97316",
        "Extra Trees Regressor": "#7C3AED",
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
    model_band = future_forecasts[MODEL_NAMES]
    model_low = model_band.min(axis=1)
    model_high = model_band.max(axis=1)
    figure.add_trace(
        go.Scatter(
            x=model_low.index,
            y=model_low,
            mode="lines",
            name="Model diapazoni",
            line=dict(color="rgba(15,118,110,0)", width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=model_high.index,
            y=model_high,
            mode="lines",
            name="Model diapazoni",
            fill="tonexty",
            fillcolor="rgba(15,118,110,0.11)",
            line=dict(color="rgba(15,118,110,0)", width=0),
            hoverinfo="skip",
        )
    )
    for model_name in ALL_FORECAST_NAMES:
        is_best = model_name == best_model_name
        is_baseline = model_name == BASELINE_NAME
        figure.add_trace(
            go.Scatter(
                x=future_forecasts.index,
                y=future_forecasts[model_name],
                mode="lines",
                name=f"{model_name}{' ★' if is_best else ''}",
                line=dict(
                    color=colors[model_name],
                    width=3.2 if is_best else 2.2 if is_baseline else 1.7,
                    dash=(
                        "dot"
                        if is_baseline and not is_best
                        else "solid"
                        if is_best
                        else "dash"
                    ),
                ),
                opacity=1.0 if is_best or is_baseline else 0.72,
                marker=dict(size=5),
            )
        )
    return apply_chart_style(figure, "Kelajak prognozi")


def inject_custom_css() -> None:
    """Streamlit interfeysiga professional, ixcham dizayn beradi."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #0F172A;
            --muted: #475569;
            --card: #FFFFFF;
            --border: #DCE3EC;
            --surface: #F6F8FB;
            --accent: #0F766E;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--surface);
        }
        .block-container {
            max-width: 1280px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        .page-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }
        .eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .page-header h1 {
            margin: 0;
            color: var(--ink);
            font-size: 2rem;
            line-height: 1.05;
        }
        .header-note {
            color: var(--muted);
            font-size: 0.86rem;
            text-align: right;
        }
        .control-panel {
            padding: 0.8rem 1rem 0.35rem;
            border-radius: 8px;
            background: var(--card);
            border: 1px solid var(--border);
            margin-bottom: 1rem;
        }
        .section-title {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 700;
            margin: 0.35rem 0 0.18rem 0;
        }
        .section-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.45rem;
        }
        .summary-panel {
            min-height: 100%;
            padding: 1rem 1.05rem;
            border-radius: 8px;
            background: var(--card);
            border: 1px solid var(--border);
        }
        .summary-kicker {
            color: var(--muted);
            font-size: 0.78rem;
            margin-bottom: 0.2rem;
        }
        .summary-price {
            color: var(--ink);
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .summary-change {
            color: #15803D;
            font-size: 0.92rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }
        .summary-change.down {
            color: #B91C1C;
        }
        .summary-divider {
            height: 1px;
            background: var(--border);
            margin: 0.9rem 0;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .summary-label {
            color: var(--muted);
            font-size: 0.76rem;
            margin-bottom: 0.16rem;
        }
        .summary-value {
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 600;
            line-height: 1.35;
        }
        .signal-pill {
            display: inline-block;
            margin-top: 0.8rem;
            padding: 0.24rem 0.55rem;
            border-radius: 999px;
            background: #E6F4F1;
            color: #0F766E;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .signal-pill.neutral {
            background: #EEF2F7;
            color: #475569;
        }
        .table-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }
        .signal-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.9rem 0 1.1rem;
        }
        .signal-card {
            padding: 0.8rem 0.9rem;
            border-radius: 8px;
            background: var(--card);
            border: 1px solid var(--border);
        }
        .signal-card-label {
            color: var(--muted);
            font-size: 0.76rem;
            margin-bottom: 0.16rem;
        }
        .signal-card-value {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.3;
        }
        .signal-card-note {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.18rem;
        }
        .workflow-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.2rem 0 1rem;
        }
        .workflow-card {
            padding: 0.8rem 0.9rem;
            border-radius: 8px;
            background: #FBFCFD;
            border: 1px solid var(--border);
        }
        .workflow-step {
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }
        .workflow-value {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.35;
        }
        .workflow-note {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.18rem;
        }
        .selection-note {
            padding: 0.8rem 0.9rem;
            margin-top: 0.7rem;
            border-radius: 8px;
            background: #F8FAFC;
            border: 1px solid var(--border);
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }
        .confidence-panel {
            padding: 1rem 1.05rem;
            border-radius: 8px;
            background: var(--card);
            border: 1px solid var(--border);
            margin: 0.15rem 0 1.05rem;
        }
        .confidence-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.85rem;
        }
        .confidence-kicker {
            color: var(--muted);
            font-size: 0.78rem;
            margin-bottom: 0.16rem;
        }
        .confidence-title {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 700;
        }
        .confidence-badge {
            display: inline-block;
            padding: 0.22rem 0.52rem;
            border-radius: 999px;
            background: #FEF3C7;
            color: #B45309;
            font-size: 0.76rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .confidence-badge.good {
            background: #DCFCE7;
            color: #15803D;
        }
        .confidence-badge.weak {
            background: #FEE2E2;
            color: #B91C1C;
        }
        .confidence-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .confidence-item {
            padding-top: 0.72rem;
            border-top: 1px solid var(--border);
        }
        .confidence-label {
            color: var(--muted);
            font-size: 0.76rem;
            margin-bottom: 0.16rem;
        }
        .confidence-value {
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 700;
        }
        .confidence-note {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.18rem;
        }
        div[data-testid="stDataFrame"] {
            font-size: 0.85rem;
        }
        div[data-testid="stDownloadButton"] button {
            width: 100%;
        }
        details {
            border-radius: 8px;
        }
        @media (max-width: 900px) {
            .page-header {
                display: block;
            }
            .header-note {
                margin-top: 0.35rem;
                text-align: left;
            }
            .summary-grid {
                grid-template-columns: 1fr;
            }
            [data-testid="stHorizontalBlock"] {
                flex-direction: column;
                gap: 0.8rem;
            }
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
            .summary-panel {
                min-width: 0;
            }
            .signal-strip {
                grid-template-columns: 1fr;
            }
            .workflow-strip {
                grid-template-columns: 1fr;
            }
            .confidence-header {
                display: block;
            }
            .confidence-badge {
                margin-top: 0.45rem;
            }
            .confidence-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(ticker: str, history_years: int) -> None:
    """Sahifa tepasidagi ishchi sarlavhani chiqaradi."""
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <div class="eyebrow">Aksiya prognoz paneli</div>
                <h1>{ticker}</h1>
            </div>
            <div class="header-note">{history_years} yillik tarix · benchmark · rolling backtest</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_panel(
    latest_close: float,
    daily_delta: float,
    daily_delta_pct: float,
    trailing_low: float,
    trailing_high: float,
    recent_volatility: float,
    best_future_model_name: str,
    best_future_skill: float,
    forecast_months: int,
    best_forecast_end: float,
    forecast_delta: float,
) -> None:
    """Asosiy natijalarni bitta ixcham panelda ko'rsatadi."""
    change_class = "down" if daily_delta < 0 else ""
    forecast_note = (
        "Benchmark yetakchi"
        if best_future_model_name == BASELINE_NAME
        else f"Benchmarkdan {best_future_skill:+.1f}%"
    )
    pill_class = "neutral" if best_future_model_name == BASELINE_NAME else ""
    st.markdown(
        f"""
        <div class="summary-panel">
            <div class="summary-kicker">Hozirgi narx</div>
            <div class="summary-price">${latest_close:,.2f}</div>
            <div class="summary-change {change_class}">
                {daily_delta:+.2f} ({daily_delta_pct:+.2f}%)
            </div>
            <div class="summary-divider"></div>
            <div class="summary-grid">
                <div>
                    <div class="summary-label">52 haftalik oralig'</div>
                    <div class="summary-value">${trailing_low:,.0f} - ${trailing_high:,.0f}</div>
                </div>
                <div>
                    <div class="summary-label">3 oylik volatillik</div>
                    <div class="summary-value">{recent_volatility:.1f}%</div>
                </div>
                <div>
                    <div class="summary-label">Tanlangan prognoz</div>
                    <div class="summary-value">{format_model_label(best_future_model_name)}</div>
                </div>
                <div>
                    <div class="summary-label">{forecast_months} oy yakuni</div>
                    <div class="summary-value">${best_forecast_end:,.2f}</div>
                </div>
            </div>
            <div class="signal-pill {pill_class}">
                {forecast_note} · Yakuniy farq {forecast_delta:+.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_strip(
    trailing_year_return: float,
    ema_20: float,
    ema_50: float,
    forecast_up_count: int,
    total_models: int,
    forecast_range_low: float,
    forecast_range_high: float,
    forecast_spread_pct: float,
) -> None:
    """Bozor va forecast bo'yicha ixcham yordamchi signallarni ko'rsatadi."""
    trend_label = "Yuqoriga" if ema_20 >= ema_50 else "Pastga"
    trend_note = f"EMA20 ${ema_20:,.2f} / EMA50 ${ema_50:,.2f}"
    st.markdown(
        f"""
        <div class="signal-strip">
            <div class="signal-card">
                <div class="signal-card-label">1 yillik o'sish</div>
                <div class="signal-card-value">{trailing_year_return:+.1f}%</div>
                <div class="signal-card-note">Oxirgi 252 savdo kuni</div>
            </div>
            <div class="signal-card">
                <div class="signal-card-label">Trend</div>
                <div class="signal-card-value">{trend_label}</div>
                <div class="signal-card-note">{trend_note}</div>
            </div>
            <div class="signal-card">
                <div class="signal-card-label">Model kelishuvi</div>
                <div class="signal-card-value">{forecast_up_count}/{total_models} yuqoriga</div>
                <div class="signal-card-note">Kelajak yakuniy nuqtasi bo'yicha</div>
            </div>
            <div class="signal-card">
                <div class="signal-card-label">Prognoz oralig'i</div>
                <div class="signal-card-value">${forecast_range_low:,.2f} - ${forecast_range_high:,.2f}</div>
                <div class="signal-card-note">Tarqoqlik {forecast_spread_pct:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_strip(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Model oqimini foydalanuvchiga yashirmasdan ko'rsatadi."""
    st.markdown(
        f"""
        <div class="workflow-strip">
            <div class="workflow-card">
                <div class="workflow-step">1. O'qitish</div>
                <div class="workflow-value">{len(train_data):,} qator</div>
                <div class="workflow-note">{train_data.index.min():%Y-%m-%d} - {train_data.index.max():%Y-%m-%d}</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-step">2. Test</div>
                <div class="workflow-value">{len(test_data):,} qator</div>
                <div class="workflow-note">{test_data.index.min():%Y-%m-%d} - {test_data.index.max():%Y-%m-%d}</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-step">3. Feature</div>
                <div class="workflow-value">{len(FEATURE_COLUMNS)} ta signal</div>
                <div class="workflow-note">Lag, rolling, return, momentum, EMA</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-step">4. Tanlov</div>
                <div class="workflow-value">{ROLLING_BACKTEST_ORIGINS} oynali backtest</div>
                <div class="workflow-note">Benchmark bilan tekshiriladi</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_panel(
    best_future_skill: float,
    best_future_benchmark_wins: int,
    best_future_wins: int,
    forecast_spread_pct: float,
    forecast_up_count: int,
    total_models: int,
    recent_volatility: float,
) -> None:
    """Forecast natijasini o'qishga yordam beradigan ishonchlilik paneli."""
    agreement_ratio = forecast_up_count / total_models if total_models else 0.0
    score = 0
    if best_future_skill > 0:
        score += 1
    if best_future_benchmark_wins >= max(1, ROLLING_BACKTEST_ORIGINS // 2):
        score += 1
    if forecast_spread_pct <= 5:
        score += 1
    if agreement_ratio >= 0.75:
        score += 1
    if recent_volatility < 35:
        score += 1

    if score >= 4:
        confidence_label = "Yaxshi"
        confidence_class = "good"
    elif score >= 2:
        confidence_label = "Ehtiyotkor"
        confidence_class = ""
    else:
        confidence_label = "Zaif"
        confidence_class = "weak"

    benchmark_note = (
        f"{best_future_benchmark_wins}/{ROLLING_BACKTEST_ORIGINS} oynada benchmarkdan yaxshi"
    )
    st.markdown(
        f"""
        <div class="confidence-panel">
            <div class="confidence-header">
                <div>
                    <div class="confidence-kicker">Prognozni o'qish</div>
                    <div class="confidence-title">Ishonchlilik va risk</div>
                </div>
                <div class="confidence-badge {confidence_class}">{confidence_label}</div>
            </div>
            <div class="confidence-grid">
                <div class="confidence-item">
                    <div class="confidence-label">Benchmarkdan ustunlik</div>
                    <div class="confidence-value">{best_future_skill:+.1f}%</div>
                    <div class="confidence-note">{benchmark_note}</div>
                </div>
                <div class="confidence-item">
                    <div class="confidence-label">G'olib oynalar</div>
                    <div class="confidence-value">{best_future_wins}/{ROLLING_BACKTEST_ORIGINS}</div>
                    <div class="confidence-note">Rolling backtest bo'yicha</div>
                </div>
                <div class="confidence-item">
                    <div class="confidence-label">Model tarqoqligi</div>
                    <div class="confidence-value">{forecast_spread_pct:.1f}%</div>
                    <div class="confidence-note">Past bo'lsa, prognozlar yaqinroq</div>
                </div>
                <div class="confidence-item">
                    <div class="confidence-label">Yo'nalish kelishuvi</div>
                    <div class="confidence-value">{forecast_up_count}/{total_models}</div>
                    <div class="confidence-note">Modellar yuqoriga deydi</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_model_label(model_name: str) -> str:
    """UI jadvalida model nomlarini ixchamroq ko'rsatadi."""
    return {
        BASELINE_NAME: "Baseline",
        "Ridge Regression": "Ridge",
        "Random Forest Regressor": "Random Forest",
        "Gradient Boosting Regressor": "Gradient Boosting",
        "Extra Trees Regressor": "Extra Trees",
    }.get(model_name, model_name)


def main() -> None:
    """Streamlit interfeysini ishga tushiradi."""
    st.set_page_config(
        page_title="Aksiya prognoz paneli",
        page_icon="📊",
        layout="wide",
    )
    inject_custom_css()

    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    control_columns = st.columns([1.0, 1.0, 0.95, 0.9])
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
        history_years = st.selectbox(
            "Tarix chuqurligi",
            options=HISTORY_YEARS,
            index=HISTORY_YEARS.index(DEFAULT_HISTORY_YEARS),
            format_func=lambda value: (
                "2 yil - tez"
                if value == 2
                else "5 yil - muvozanat"
                if value == 5
                else "10 yil - to'liq"
            ),
        )
    with control_columns[3]:
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

    with st.spinner("Real narx, tarixiy data va modellar tayyorlanmoqda..."):
        data, demo_mode = load_stock_data(ticker, history_years)
        live_quote = load_live_quote(ticker)
        train_data, test_data, metrics_table, predictions, best_model_name = (
            run_model_evaluation(data)
        )
        future_forecasts = run_all_future_forecasts(data, forecast_months)
        rolling_summary, _rolling_details = run_cached_rolling_horizon_backtests(
            data,
            forecast_months,
        )

    render_page_header(ticker, history_years)

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
    best_future_model_name = str(rolling_summary.iloc[0]["Model"])
    best_future_skill = float(rolling_summary.iloc[0]["Skill vs Benchmark"])
    best_future_benchmark_wins = int(rolling_summary.iloc[0]["Benchmark Wins"])
    best_future_wins = int(rolling_summary.iloc[0]["Wins"])
    best_forecast_end = float(future_forecasts[best_future_model_name].iloc[-1])
    forecast_anchor = float(data["Close"].iloc[-1])
    forecast_delta = best_forecast_end - forecast_anchor
    trailing_year = data.tail(252)
    trailing_low = float(trailing_year["Low"].min())
    trailing_high = float(trailing_year["High"].max())
    recent_volatility = float(
        data["Close"].pct_change().tail(63).std() * np.sqrt(252) * 100
    )
    trailing_year_return = (
        (float(data["Close"].iloc[-1]) / float(data["Close"].iloc[-252])) - 1
    ) * 100
    ema_20 = float(data["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    ema_50 = float(data["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
    forecast_end_values = future_forecasts[MODEL_NAMES].iloc[-1]
    forecast_up_count = int((forecast_end_values > forecast_anchor).sum())
    forecast_range_low = float(forecast_end_values.min())
    forecast_range_high = float(forecast_end_values.max())
    forecast_spread_pct = (
        ((forecast_range_high - forecast_range_low) / forecast_anchor) * 100
        if forecast_anchor
        else 0.0
    )

    display_table = metrics_table.copy()
    display_table["Model"] = display_table["Model"].map(format_model_label)
    display_table.insert(0, "Rank", range(1, len(display_table) + 1))
    display_table["Holat"] = display_table["Model"].map(
        lambda model_name: (
            "Eng yaxshi" if model_name == format_model_label(best_model_name) else ""
        )
    )
    for column in ["MAE", "RMSE", "MAPE", "R² Score"]:
        display_table[column] = display_table[column].map(lambda value: round(value, 4))

    if best_future_model_name == BASELINE_NAME:
        st.info(
            "Rolling backtest natijasida oddiy benchmark eng yaxshi chiqdi. "
            "Ilova buni yashirmaydi: murakkab model benchmarkdan ustun bo'lmasa, "
            "eng ishonchli signal sifatida benchmark ko'rsatiladi."
        )

    overview_left, overview_right = st.columns([1.62, 0.88], gap="large")
    with overview_left:
        st.markdown('<div class="section-title">Narx harakati</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Oxirgi 12 oy narxi, EMA 20 va EMA 50 bilan.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(create_price_chart(data, ticker), width="stretch")
    with overview_right:
        render_summary_panel(
            latest_close=latest_close,
            daily_delta=daily_delta,
            daily_delta_pct=daily_delta_pct,
            trailing_low=trailing_low,
            trailing_high=trailing_high,
            recent_volatility=recent_volatility,
            best_future_model_name=best_future_model_name,
            best_future_skill=best_future_skill,
            forecast_months=forecast_months,
            best_forecast_end=best_forecast_end,
            forecast_delta=forecast_delta,
        )

    render_signal_strip(
        trailing_year_return=trailing_year_return,
        ema_20=ema_20,
        ema_50=ema_50,
        forecast_up_count=forecast_up_count,
        total_models=len(MODEL_NAMES),
        forecast_range_low=forecast_range_low,
        forecast_range_high=forecast_range_high,
        forecast_spread_pct=forecast_spread_pct,
    )
    render_confidence_panel(
        best_future_skill=best_future_skill,
        best_future_benchmark_wins=best_future_benchmark_wins,
        best_future_wins=best_future_wins,
        forecast_spread_pct=forecast_spread_pct,
        forecast_up_count=forecast_up_count,
        total_models=len(MODEL_NAMES),
        recent_volatility=recent_volatility,
    )
    st.markdown('<div class="section-title">Model oqimi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Qaysi data bilan o‘qitildi, qayerda test qilindi va tanlov qanday qilindi.</div>',
        unsafe_allow_html=True,
    )
    render_workflow_strip(train_data=train_data, test_data=test_data)

    test_left, test_right = st.columns([1.62, 0.88], gap="large")
    with test_left:
        st.markdown(
            '<div class="section-title">Model test natijalari</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-note">Oxirgi 1 yillik holdout test: haqiqiy narx, benchmark va barcha model chiziqlari bitta grafikda.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            create_backtest_chart(
                actual=test_data.loc[predictions[best_model_name].index, "Close"],
                predictions=predictions,
                best_model_name=best_model_name,
            ),
            width="stretch",
        )
    with test_right:
        st.markdown('<div class="section-title">Test jadvali</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">MAE, RMSE, MAPE va R² bo‘yicha kunlik test natijasi.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(display_table, width="stretch", hide_index=True)
        st.markdown(
            '<div class="table-note">RMSE kichikroq va R² kattaroq bo‘lsa, model testda yaxshiroq ishlagan.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="selection-note">
                Kunlik test yetakchisi: <strong>{format_model_label(best_model_name)}</strong><br>
                Kelajak tanlovi: <strong>{format_model_label(best_future_model_name)}</strong><br>
                Ular farq qilishi mumkin, chunki birinchisi kunlik holdout test, ikkinchisi esa kelajak ufqiga mos rolling backtest.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Holdout test CSV",
            data=display_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_holdout_test.csv",
            mime="text/csv",
        )

    st.markdown('<div class="section-title">Prognoz va sifat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Asosiy tanlov bir martalik split emas, rolling backtest oynalari o‘rtachasi bo‘yicha qilinadi.</div>',
        unsafe_allow_html=True,
    )
    horizon_display = rolling_summary[
        [
            "Model",
            "RMSE",
            "MAPE",
            "Skill vs Benchmark",
            "Benchmark Wins",
            "Wins",
        ]
    ].copy()
    horizon_display["Model"] = horizon_display["Model"].map(format_model_label)
    horizon_display.insert(0, "Rank", range(1, len(horizon_display) + 1))
    horizon_display = horizon_display.rename(
        columns={
            "Benchmark Wins": "Benchmarkdan yaxshi",
            "Wins": "G'olib oynalar",
        }
    )
    for column in ["RMSE", "MAPE", "Skill vs Benchmark"]:
        horizon_display[column] = horizon_display[column].map(lambda value: round(value, 4))
    analysis_left, analysis_right = st.columns([1.62, 0.88], gap="large")
    with analysis_left:
        st.markdown('<div class="section-title">Kelajak prognozi</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Benchmark va 4 ta model bir xil tarixiy featurelar asosida solishtiriladi.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            create_multi_model_forecast_chart(data, future_forecasts, best_future_model_name),
            width="stretch",
        )
    with analysis_right:
        st.markdown('<div class="section-title">Model sifati</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Rolling backtest bo‘yicha qisqa reyting.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(horizon_display, width="stretch", hide_index=True)
        st.markdown(
            '<div class="table-note">RMSE qancha kichik bo‘lsa, prognoz shuncha yaxshi.</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Rolling backtest CSV",
            data=rolling_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_{forecast_months}oy_rolling_backtest.csv",
            mime="text/csv",
        )

    with st.expander("Qo‘shimcha diagnostika"):
        with st.spinner("Qo'shimcha tekshiruvlar hisoblanmoqda..."):
            horizon_actual, horizon_forecasts = run_cached_horizon_backtest(
                data,
                forecast_months,
            )
        st.markdown("#### Train/test kesimi")
        left_column, right_column = st.columns(2)
        with left_column:
            st.plotly_chart(
                create_train_test_chart(train_data, test_data),
                width="stretch",
            )
        with right_column:
            selected_actual = test_data.loc[predictions[best_model_name].index, "Close"]
            st.plotly_chart(
                create_prediction_chart(
                    actual=selected_actual,
                    predicted=predictions[best_model_name],
                    model_name=best_model_name,
                ),
                width="stretch",
            )
        st.markdown("#### Oxirgi horizon-matched backtest")
        st.caption(
            "Bu chart kelajak forecast bilan bir xil ufqda oxirgi tarixiy oynani ko‘rsatadi."
        )
        st.plotly_chart(
            create_horizon_backtest_chart(horizon_actual, horizon_forecasts),
            width="stretch",
        )
        st.markdown("#### Klassik ML uchun walk-forward validation")
        st.caption(
            "Kunlik model sifati vaqt bo‘yicha qanchalik barqarorligini ko‘rsatadi."
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
            st.dataframe(walk_forward_display, width="stretch", hide_index=True)

    st.download_button(
        "Prognozlar CSV",
        data=future_forecasts.reset_index(names="Date").to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_{forecast_months}oy_forecast.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

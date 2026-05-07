import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "momentum_20d",
    "sma_ratio_10_25",
    "sma_ratio_20_50",
    "price_vs_sma10",
    "price_vs_sma25",
    "ema_gap",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "volatility_10d",
    "volatility_20d",
    "atr_pct",
    "bollinger_position",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "hl_range",
    "oc_change",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["return_1d"] = data["close"].pct_change(1)
    data["return_3d"] = data["close"].pct_change(3)
    data["return_5d"] = data["close"].pct_change(5)
    data["return_10d"] = data["close"].pct_change(10)
    data["momentum_20d"] = data["close"] / data["close"].shift(20) - 1

    data["sma_10"] = data["close"].rolling(10).mean()
    data["sma_20"] = data["close"].rolling(20).mean()
    data["sma_25"] = data["close"].rolling(25).mean()
    data["sma_50"] = data["close"].rolling(50).mean()

    data["sma_ratio_10_25"] = data["sma_10"] / data["sma_25"]
    data["sma_ratio_20_50"] = data["sma_20"] / data["sma_50"]
    data["price_vs_sma10"] = (data["close"] - data["sma_10"]) / data["sma_10"]
    data["price_vs_sma25"] = (data["close"] - data["sma_25"]) / data["sma_25"]

    ema_12 = data["close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["close"].ewm(span=26, adjust=False).mean()
    data["ema_gap"] = (ema_12 - ema_26) / data["close"]
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    delta = data["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    data["volatility_10d"] = data["return_1d"].rolling(10).std()
    data["volatility_20d"] = data["return_1d"].rolling(20).std()

    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr_14"] = tr.rolling(14).mean()
    data["atr_pct"] = data["atr_14"] / data["close"]

    rolling_std_20 = data["close"].rolling(20).std()
    upper_band = data["sma_20"] + 2 * rolling_std_20
    lower_band = data["sma_20"] - 2 * rolling_std_20
    band_range = (upper_band - lower_band).replace(0, np.nan)
    data["bollinger_position"] = (data["close"] - lower_band) / band_range

    data["volume_ratio_5d"] = data["volume"] / data["volume"].rolling(5).mean()
    data["volume_ratio_20d"] = data["volume"] / data["volume"].rolling(20).mean()
    data["hl_range"] = (data["high"] - data["low"]) / data["close"]
    data["oc_change"] = (data["close"] - data["open"]) / data["open"]

    data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)
    data["next_return"] = data["close"].shift(-1) / data["close"] - 1
    return data


def _build_models():
    return {
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=7,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
    }


def _feature_importance_map(model_name: str, fitted_pipeline: Pipeline) -> dict:
    estimator = fitted_pipeline.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
    else:
        values = np.zeros(len(FEATURE_COLS), dtype=float)

    total = values.sum()
    if total > 0:
        values = values / total

    return {feature: float(value) for feature, value in zip(FEATURE_COLS, values)}


def _aggregate_feature_importances(importances: list[dict]) -> dict:
    combined = {feature: 0.0 for feature in FEATURE_COLS}
    for feature_map in importances:
        for feature, value in feature_map.items():
            combined[feature] += value

    total = sum(combined.values())
    if total > 0:
        combined = {feature: value / total for feature, value in combined.items()}

    ordered = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    return {feature: round(value, 4) for feature, value in ordered[:10]}


def _signal_label(prob_up: float) -> str:
    edge = abs(prob_up - 0.5)
    if edge < 0.03:
        return "HOLD"
    return "UP" if prob_up >= 0.5 else "DOWN"


def _signal_strength(prob_up: float) -> str:
    edge = abs(prob_up - 0.5)
    if edge >= 0.2:
        return "Strong"
    if edge >= 0.1:
        return "Moderate"
    if edge >= 0.03:
        return "Weak"
    return "Neutral"


def _market_regime(latest_row: pd.Series) -> str:
    if latest_row["sma_ratio_20_50"] > 1 and latest_row["rsi_14"] >= 55:
        return "Bullish trend"
    if latest_row["sma_ratio_20_50"] < 1 and latest_row["rsi_14"] <= 45:
        return "Bearish trend"
    return "Range bound"


def _risk_level(latest_row: pd.Series) -> str:
    vol = latest_row["volatility_20d"]
    atr = latest_row["atr_pct"]
    if vol >= 0.03 or atr >= 0.04:
        return "High"
    if vol >= 0.02 or atr >= 0.025:
        return "Medium"
    return "Low"


def train_and_predict(df: pd.DataFrame) -> dict:
    data = compute_features(df)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 220:
        raise ValueError(
            f"Only {len(data)} usable rows after feature engineering. Need at least 220 bars for reliable time-series validation."
        )

    train_df = data.iloc[:-1].copy()
    latest_row = data.iloc[-1].copy()

    X = train_df[FEATURE_COLS]
    y = train_df["target"]

    folds = min(5, max(3, len(train_df) // 60))
    splitter = TimeSeriesSplit(n_splits=folds)

    validation_probabilities = pd.Series(index=train_df.index, dtype=float)
    fold_details = []

    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        fold_model_probs = []
        for model in _build_models().values():
            model.fit(X_train, y_train)
            fold_model_probs.append(model.predict_proba(X_val)[:, 1])

        ensemble_val_prob = np.mean(np.vstack(fold_model_probs), axis=0)
        validation_probabilities.iloc[val_idx] = ensemble_val_prob

        fold_predictions = (ensemble_val_prob >= 0.5).astype(int)
        fold_details.append(
            {
                "fold": fold_number,
                "train_samples": int(len(train_idx)),
                "validation_samples": int(len(val_idx)),
                "accuracy": round(float(accuracy_score(y_val, fold_predictions)) * 100, 2),
            }
        )

    valid_mask = validation_probabilities.notna()
    y_valid = y.loc[valid_mask]
    prob_valid = validation_probabilities.loc[valid_mask]
    pred_valid = (prob_valid >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_valid, pred_valid)) * 100, 2),
        "precision": round(float(precision_score(y_valid, pred_valid, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_valid, pred_valid, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_valid, pred_valid, zero_division=0)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_valid, prob_valid)) * 100, 2) if y_valid.nunique() > 1 else None,
    }

    latest_features = latest_row[FEATURE_COLS].to_frame().T
    final_models = _build_models()
    final_importances = []
    latest_model_probs = []

    for model_name, model in final_models.items():
        model.fit(X, y)
        latest_model_probs.append(float(model.predict_proba(latest_features)[0, 1]))
        final_importances.append(_feature_importance_map(model_name, model))

    prob_up = float(np.mean(latest_model_probs))
    prob_down = 1 - prob_up
    prediction = _signal_label(prob_up)
    confidence = prob_up if prediction == "UP" else prob_down if prediction == "DOWN" else max(prob_up, prob_down)

    latest_snapshot = {
        "return_1d": round(float(latest_row["return_1d"]) * 100, 2),
        "return_5d": round(float(latest_row["return_5d"]) * 100, 2),
        "rsi_14": round(float(latest_row["rsi_14"]), 2),
        "volatility_20d": round(float(latest_row["volatility_20d"]) * 100, 2),
        "atr_pct": round(float(latest_row["atr_pct"]) * 100, 2),
        "volume_ratio_20d": round(float(latest_row["volume_ratio_20d"]), 2),
        "price_vs_sma25": round(float(latest_row["price_vs_sma25"]) * 100, 2),
        "macd_hist": round(float(latest_row["macd_hist"]), 4),
    }

    return {
        "prediction": prediction,
        "confidence": round(float(confidence) * 100, 2),
        "up_probability": round(prob_up * 100, 2),
        "down_probability": round(prob_down * 100, 2),
        "signal_strength": _signal_strength(prob_up),
        "market_regime": _market_regime(latest_row),
        "risk_level": _risk_level(latest_row),
        "model_accuracy": metrics["accuracy"],
        "metrics": metrics,
        "feature_importances": _aggregate_feature_importances(final_importances),
        "fold_details": fold_details,
        "training_samples": int(len(X)),
        "validation_samples": int(valid_mask.sum()),
        "latest_feature_snapshot": latest_snapshot,
        "model_stack": ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        "summary": (
            f"Ensemble models indicate a {prediction} bias with {round(float(confidence) * 100, 2)}% confidence. "
            f"Current regime is { _market_regime(latest_row).lower() } with { _risk_level(latest_row).lower() } risk conditions."
        ),
    }


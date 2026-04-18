import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers technical indicator features from raw OHLCV data.
    All features are ratio/percentage-based so they're scale-invariant across symbols.
    """
    df = df.copy()

    # --- Momentum / Return features ---
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)

    # --- SMA features (reuses the same SMAs as your SmaCross strategy) ---
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_25'] = df['close'].rolling(25).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_25']  # > 1.0 = bullish regime
    df['price_vs_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
    df['price_vs_sma25'] = (df['close'] - df['sma_25']) / df['sma_25']

    # --- Volatility ---
    df['volatility_10d'] = df['return_1d'].rolling(10).std()

    # --- RSI (14-period) ---
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)          # avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- Volume ---
    df['volume_change'] = df['volume'].pct_change(1)

    # --- Intraday range ---
    df['hl_range'] = (df['high'] - df['low']) / df['close']

    # --- Target label: 1 if tomorrow's close > today's close, else 0 ---
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    return df


FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_10d',
    'sma_ratio', 'price_vs_sma10', 'price_vs_sma25',
    'volatility_10d', 'rsi', 'volume_change', 'hl_range'
]


def train_and_predict(df: pd.DataFrame) -> dict:
    """
    Trains a RandomForest on historical bars and predicts direction for the latest bar.

    Returns a dict with:
      - prediction       : "UP" or "DOWN"
      - confidence       : probability of the predicted class (%)
      - model_accuracy   : accuracy on held-out test set (%)
      - feature_importances : ranked dict of which features mattered most
      - training_samples / test_samples : data split sizes
    """
    df = compute_features(df)
    df = df.dropna()          # drop rows where rolling windows haven't filled yet

    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} usable rows after feature engineering. "
            "Need at least 60 bars. Try a longer date range."
        )

    # The last row has no next-day target yet (market hasn't closed), so:
    #   - train/test on all rows EXCEPT the last
    #   - predict on the last row
    train_df = df.iloc[:-1]
    predict_row = df.iloc[[-1]]

    X = train_df[FEATURE_COLS]
    y = train_df['target']

    # Chronological split — no shuffle, because time order matters
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,   # prevents overfitting on small datasets
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on unseen (future) test set
    y_pred   = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Predict the next move for the latest bar
    X_latest      = scaler.transform(predict_row[FEATURE_COLS])
    prediction    = model.predict(X_latest)[0]           # 0 or 1
    probabilities = model.predict_proba(X_latest)[0]     # [prob_down, prob_up]
    confidence    = float(max(probabilities))

    # Feature importances — useful for the frontend to show "why"
    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    sorted_importances = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "prediction":             "UP"   if prediction == 1 else "DOWN",
        "confidence":             round(confidence * 100, 2),
        "model_accuracy":         round(accuracy * 100, 2),
        "feature_importances":    sorted_importances,
        "training_samples":       len(X_train),
        "test_samples":           len(X_test),
    }

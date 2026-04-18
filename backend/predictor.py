import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    'return_1d',
    'return_2d',
    'return_3d',
    'return_5d',
    'return_10d',
    'return_20d',
    'volatility_5d',
    'volatility_10d',
    'volatility_20d',
    'price_vs_sma10',
    'price_vs_sma20',
    'price_vs_sma50',
    'sma10_vs_sma20',
    'sma20_vs_sma50',
    'ema12_vs_ema26',
    'rsi_14',
    'macd',
    'macd_signal',
    'macd_hist',
    'atr_14_pct',
    'bb_width',
    'bb_zscore',
    'volume_change',
    'volume_z20',
    'obv_slope_10',
    'hl_range',
    'close_location',
    'gap_open',
    'streak_5'
]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).lower() for col in df.columns]
    df = df.sort_index()

    close = df['close'].astype(float)
    open_ = df['open'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float).replace(0, np.nan)

    df['return_1d'] = close.pct_change(1)
    df['return_2d'] = close.pct_change(2)
    df['return_3d'] = close.pct_change(3)
    df['return_5d'] = close.pct_change(5)
    df['return_10d'] = close.pct_change(10)
    df['return_20d'] = close.pct_change(20)

    df['volatility_5d'] = df['return_1d'].rolling(5).std()
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()

    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df['price_vs_sma10'] = (close - sma10) / (sma10 + 1e-9)
    df['price_vs_sma20'] = (close - sma20) / (sma20 + 1e-9)
    df['price_vs_sma50'] = (close - sma50) / (sma50 + 1e-9)
    df['sma10_vs_sma20'] = (sma10 - sma20) / (sma20 + 1e-9)
    df['sma20_vs_sma50'] = (sma20 - sma50) / (sma50 + 1e-9)
    df['ema12_vs_ema26'] = (ema12 - ema26) / (ema26 + 1e-9)

    df['rsi_14'] = _rsi(close, 14)

    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    atr14 = _atr(df, 14)
    df['atr_14_pct'] = atr14 / (close + 1e-9)

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    df['bb_width'] = (upper_band - lower_band) / (rolling_mean + 1e-9)
    df['bb_zscore'] = (close - rolling_mean) / (rolling_std + 1e-9)

    df['volume_change'] = volume.pct_change()
    log_volume = np.log1p(volume)
    df['volume_z20'] = (log_volume - log_volume.rolling(20).mean()) / (log_volume.rolling(20).std() + 1e-9)

    signed_volume = np.sign(close.diff().fillna(0)) * volume.fillna(0)
    obv = signed_volume.cumsum()
    df['obv_slope_10'] = obv.diff(10) / (obv.abs().rolling(20).mean() + 1)

    df['hl_range'] = (high - low) / (close + 1e-9)
    df['close_location'] = ((close - low) / ((high - low) + 1e-9)).clip(0, 1)
    df['gap_open'] = (open_ - close.shift(1)) / (close.shift(1) + 1e-9)
    df['streak_5'] = np.sign(df['return_1d']).rolling(5).sum()

    df['target'] = (close.shift(-1) > close).astype(int)
    return df


def _build_models():
    return {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, class_weight='balanced'))
        ]),
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }


def _evaluate_models(X: pd.DataFrame, y: pd.Series):
    suggested_splits = max(3, min(5, len(X) // 120))
    n_splits = min(suggested_splits, len(X) - 1)
    if n_splits < 2:
        raise ValueError('Not enough data to run walk-forward validation.')

    splitter = TimeSeriesSplit(n_splits=n_splits)
    models = _build_models()
    scores = {name: {'accuracy': [], 'balanced_accuracy': [], 'f1': [], 'precision_up': [], 'recall_up': []} for name in models}

    for train_idx, valid_idx in splitter.split(X):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        if y_train.nunique() < 2 or y_valid.nunique() < 2:
            continue

        for name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)
            pred = fitted_model.predict(X_valid)

            scores[name]['accuracy'].append(accuracy_score(y_valid, pred))
            scores[name]['balanced_accuracy'].append(balanced_accuracy_score(y_valid, pred))
            scores[name]['f1'].append(f1_score(y_valid, pred, zero_division=0))
            scores[name]['precision_up'].append(precision_score(y_valid, pred, zero_division=0))
            scores[name]['recall_up'].append(recall_score(y_valid, pred, zero_division=0))

    leaderboard = []
    for name, metric_values in scores.items():
        if not metric_values['accuracy']:
            continue
        leaderboard.append({
            'name': name,
            'accuracy': np.mean(metric_values['accuracy']),
            'balanced_accuracy': np.mean(metric_values['balanced_accuracy']),
            'f1': np.mean(metric_values['f1']),
            'precision_up': np.mean(metric_values['precision_up']),
            'recall_up': np.mean(metric_values['recall_up'])
        })

    if not leaderboard:
        raise ValueError('Unable to score models on the available training data.')

    leaderboard.sort(key=lambda row: (row['balanced_accuracy'], row['accuracy'], row['f1']), reverse=True)
    return leaderboard, n_splits


def _fit_ensemble(X: pd.DataFrame, y: pd.Series, leaderboard: list):
    selected_rows = leaderboard[: min(3, len(leaderboard))]
    weights = np.array([max(row['balanced_accuracy'], 0.01) for row in selected_rows], dtype=float)
    weights = weights / weights.sum()

    models = _build_models()
    fitted_models = []
    for row, weight in zip(selected_rows, weights):
        model = clone(models[row['name']])
        model.fit(X, y)
        fitted_models.append((row['name'], model, float(weight)))

    return fitted_models, selected_rows


def _extract_feature_strengths(fitted_models: list):
    aggregate = pd.Series(0.0, index=FEATURE_COLS)

    for _, model, weight in fitted_models:
        estimator = model.named_steps['model'] if hasattr(model, 'named_steps') else model
        raw_strength = None

        if hasattr(estimator, 'feature_importances_'):
            raw_strength = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, 'coef_'):
            raw_strength = np.abs(np.asarray(estimator.coef_[0], dtype=float))

        if raw_strength is None or raw_strength.sum() == 0:
            continue

        normalized = raw_strength / raw_strength.sum()
        aggregate += normalized * weight

    aggregate = aggregate.sort_values(ascending=False)
    return [
        {'feature': feature, 'importance': round(float(score) * 100, 2)}
        for feature, score in aggregate.head(8).items()
    ]


def _market_regime(latest_row: pd.Series) -> str:
    if latest_row['sma20_vs_sma50'] > 0 and latest_row['rsi_14'] >= 55:
        return 'Bullish'
    if latest_row['sma20_vs_sma50'] < 0 and latest_row['rsi_14'] <= 45:
        return 'Bearish'
    return 'Sideways'


def train_and_predict(df: pd.DataFrame) -> dict:
    enriched = compute_features(df).replace([np.inf, -np.inf], np.nan).dropna()

    if len(enriched) < 250:
        raise ValueError(
            f'Only {len(enriched)} usable rows after feature engineering. Need at least 250 bars for stable walk-forward validation.'
        )

    train_df = enriched.iloc[:-1].copy()
    predict_row = enriched.iloc[[-1]].copy()

    X = train_df[FEATURE_COLS]
    y = train_df['target'].astype(int)

    if y.nunique() < 2:
        raise ValueError('The training target contains only one class. Try a longer date range or a different symbol.')

    leaderboard, cv_folds = _evaluate_models(X, y)
    fitted_models, selected_rows = _fit_ensemble(X, y, leaderboard)

    latest_features = predict_row[FEATURE_COLS]
    weighted_prob_up = 0.0
    for _, model, weight in fitted_models:
        prob_up = float(model.predict_proba(latest_features)[0][1])
        weighted_prob_up += prob_up * weight

    weighted_prob_down = 1 - weighted_prob_up
    prediction = 'UP' if weighted_prob_up >= 0.5 else 'DOWN'
    confidence = max(weighted_prob_up, weighted_prob_down)

    top_row = selected_rows[0]
    feature_importances = _extract_feature_strengths(fitted_models)
    latest = predict_row.iloc[0]

    if weighted_prob_up >= 0.58:
        recommendation = 'BUY'
    elif weighted_prob_up <= 0.42:
        recommendation = 'SELL'
    else:
        recommendation = 'HOLD'

    return {
        'prediction': prediction,
        'recommendation': recommendation,
        'confidence': round(confidence * 100, 2),
        'probability_up': round(weighted_prob_up * 100, 2),
        'probability_down': round(weighted_prob_down * 100, 2),
        'model_accuracy': round(top_row['accuracy'] * 100, 2),
        'balanced_accuracy': round(top_row['balanced_accuracy'] * 100, 2),
        'f1_score': round(top_row['f1'] * 100, 2),
        'precision_up': round(top_row['precision_up'] * 100, 2),
        'recall_up': round(top_row['recall_up'] * 100, 2),
        'cv_folds': cv_folds,
        'selected_models': [row['name'] for row in selected_rows],
        'feature_importances': feature_importances,
        'feature_count': len(FEATURE_COLS),
        'training_samples': int(len(train_df)),
        'market_regime': _market_regime(latest),
        'latest_indicators': {
            'rsi_14': round(float(latest['rsi_14']), 2),
            'macd_hist': round(float(latest['macd_hist']), 4),
            'price_vs_sma20_pct': round(float(latest['price_vs_sma20']) * 100, 2),
            'bb_zscore': round(float(latest['bb_zscore']), 2),
            'atr_14_pct': round(float(latest['atr_14_pct']) * 100, 2),
            'volume_z20': round(float(latest['volume_z20']), 2)
        },
        'model_leaderboard': [
            {
                'name': row['name'],
                'accuracy': round(float(row['accuracy']) * 100, 2),
                'balanced_accuracy': round(float(row['balanced_accuracy']) * 100, 2),
                'f1_score': round(float(row['f1']) * 100, 2)
            }
            for row in leaderboard
        ]
    }

import React, { useMemo, useState } from 'react';
import './PredictionPanel.css';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

function MetricCard({ label, value }) {
  return (
    <div className="prediction-metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function toLabel(text) {
  return text
    .replace(/_/g, ' ')
    .replace(/\b\w/g, letter => letter.toUpperCase());
}

function PredictionPanel({ symbol }) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';
  const cleanSymbol = symbol.trim();

  const topFeatures = useMemo(() => prediction?.feature_importances || [], [prediction]);
  const latestIndicators = useMemo(() => Object.entries(prediction?.latest_indicators || {}), [prediction]);

  const handleRunPrediction = () => {
    if (!cleanSymbol) {
      return;
    }

    setLoading(true);
    setError('');

    fetch(`${API_URL}/api/predict/${cleanSymbol}`)
      .then(async response => {
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Failed to load prediction.');
        }
        return data;
      })
      .then(data => {
        setPrediction(data);
        setLoading(false);
      })
      .catch(fetchError => {
        console.error('Prediction request failed:', fetchError);
        setPrediction(null);
        setError(fetchError.message || 'Failed to load prediction.');
        setLoading(false);
      });
  };

  return (
    <div className="prediction-container">
      <div className="prediction-header">
        <div>
          <h3>AI Stock Prediction</h3>
          <p>Walk-forward ensemble model with technical indicators, confidence score and probability split.</p>
        </div>
        <button onClick={handleRunPrediction} disabled={!cleanSymbol || loading}>
          {loading ? 'Running...' : 'Run Prediction'}
        </button>
      </div>

      {!cleanSymbol && (
        <div className="prediction-empty">
          Enter a stock symbol to see the prediction, confidence, recent price trend and model diagnostics.
        </div>
      )}

      {error && <div className="prediction-error">{error}</div>}

      {prediction && !error && (
        <>
          <div className="prediction-grid">
            <div className="prediction-highlight-card">
              <div className={`prediction-direction ${prediction.prediction === 'UP' ? 'up' : 'down'}`}>
                {prediction.prediction}
              </div>
              <h4>{prediction.symbol}</h4>
              <p className="prediction-recommendation">{prediction.recommendation}</p>
              <p className="prediction-caption">As of {prediction.as_of}</p>
            </div>

            <MetricCard label="Confidence" value={`${prediction.confidence}%`} />
            <MetricCard label="Probability Up" value={`${prediction.probability_up}%`} />
            <MetricCard label="Probability Down" value={`${prediction.probability_down}%`} />
            <MetricCard label="Model Accuracy" value={`${prediction.model_accuracy}%`} />
            <MetricCard label="Balanced Accuracy" value={`${prediction.balanced_accuracy}%`} />
            <MetricCard label="Market Regime" value={prediction.market_regime} />
            <MetricCard label="Training Samples" value={prediction.training_samples} />
          </div>

          {prediction.recent_closes?.length > 0 && (
            <div className="prediction-panel chart-panel">
              <h4>Recent Close Trend</h4>
              <div className="prediction-chart">
                <ResponsiveContainer>
                  <LineChart data={prediction.recent_closes}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" minTickGap={30} />
                    <YAxis domain={['auto', 'auto']} />
                    <Tooltip />
                    <Line type="monotone" dataKey="close" stroke="#4ac26c" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          <div className="prediction-bottom-grid">
            <div className="prediction-panel">
              <h4>Model Leaderboard</h4>
              <div className="prediction-table-header">
                <span>Model</span>
                <span>Accuracy</span>
                <span>Balanced</span>
                <span>F1</span>
              </div>
              {prediction.model_leaderboard?.map(row => (
                <div className="prediction-table-row" key={row.name}>
                  <span>{toLabel(row.name)}</span>
                  <span>{row.accuracy}%</span>
                  <span>{row.balanced_accuracy}%</span>
                  <span>{row.f1_score}%</span>
                </div>
              ))}
            </div>

            <div className="prediction-panel">
              <h4>Top Feature Drivers</h4>
              {topFeatures.map(item => (
                <div className="feature-row" key={item.feature}>
                  <div className="feature-meta">
                    <span>{toLabel(item.feature)}</span>
                    <strong>{item.importance}%</strong>
                  </div>
                  <div className="feature-bar-track">
                    <div className="feature-bar-fill" style={{ width: `${item.importance}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="prediction-bottom-grid">
            <div className="prediction-panel">
              <h4>Latest Indicators</h4>
              <div className="indicator-grid">
                {latestIndicators.map(([key, value]) => (
                  <div className="indicator-item" key={key}>
                    <span>{toLabel(key)}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </div>

            <div className="prediction-panel">
              <h4>Probability Split</h4>
              <div className="probability-row">
                <span>Up</span>
                <strong>{prediction.probability_up}%</strong>
              </div>
              <div className="feature-bar-track probability-track">
                <div className="feature-bar-fill" style={{ width: `${prediction.probability_up}%` }} />
              </div>
              <div className="probability-row probability-row-lower">
                <span>Down</span>
                <strong>{prediction.probability_down}%</strong>
              </div>
              <div className="feature-bar-track probability-track probability-track-down">
                <div className="feature-bar-fill probability-down" style={{ width: `${prediction.probability_down}%` }} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default PredictionPanel;

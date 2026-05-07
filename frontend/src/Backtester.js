import React, { useState } from 'react';
import './Backtester.css';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

function Backtester({ symbol }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [liveStatus, setLiveStatus] = useState('');
  const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

  const handleRunBacktest = () => {
    if (!symbol) {
      return;
    }

    setLoading(true);
    setResult(null);

    fetch(`${API_URL}/api/backtest/${symbol}`, { method: 'POST' })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to run backtest.');
        }

        return data;
      })
      .then((data) => setResult(data))
      .catch((error) => setResult({ error: error.message }))
      .finally(() => setLoading(false));
  };

  const handleStartLive = () => {
    if (!symbol) {
      return;
    }

    setLiveStatus(`Starting live paper trading for ${symbol}...`);

    fetch(`${API_URL}/api/livetrade/start/${symbol}`, { method: 'POST' })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to start live trading.');
        }

        return data;
      })
      .then((data) => setLiveStatus(data.message || data.detail))
      .catch((error) => setLiveStatus(`Error: ${error.message}`));
  };

  const handleStopLive = () => {
    if (!symbol) {
      return;
    }

    setLiveStatus(`Stopping live paper trading for ${symbol}...`);

    fetch(`${API_URL}/api/livetrade/stop/${symbol}`, { method: 'POST' })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to stop live trading.');
        }

        return data;
      })
      .then((data) => setLiveStatus(data.message || data.detail))
      .catch((error) => setLiveStatus(`Error: ${error.message}`));
  };

  return (
    <div className="backtester-container">
      <div className="section-header">
        <div>
          <p className="section-tag secondary-tag">Secondary Validation Layer</p>
          <h2>Backtesting and Paper Trading</h2>
        </div>
      </div>

      <p className="section-description">
        The execution layer now supports the machine learning engine. Use this section to validate how the strategy behaves historically and control paper trading actions.
      </p>

      <div className="backtester-controls">
        <button onClick={handleRunBacktest} disabled={!symbol || loading}>
          {loading ? 'Running Backtest...' : 'Run Backtest'}
        </button>
        <button onClick={handleStartLive} disabled={!symbol}>
          Start Live
        </button>
        <button onClick={handleStopLive} disabled={!symbol}>
          Stop Live
        </button>
      </div>

      {liveStatus && <div className="status-text">{liveStatus}</div>}

      {loading && <div className="loading-state">Running backtest and generating ML validation overlay...</div>}

      {result?.error && <div className="error-state">{result.error}</div>}

      {result && !result.error && (
        <div className="backtest-results">
          <div className="result-grid">
            <div className="metric-card">
              <span>Starting Value</span>
              <strong>${result.starting_value?.toLocaleString()}</strong>
            </div>
            <div className="metric-card">
              <span>Final Value</span>
              <strong>${result.final_value?.toLocaleString()}</strong>
            </div>
            <div className="metric-card">
              <span>Absolute Return</span>
              <strong>${result.absolute_return?.toLocaleString()}</strong>
            </div>
            <div className="metric-card">
              <span>Percent Return</span>
              <strong>{result.percent_return}%</strong>
            </div>
          </div>

          {result.ml_overlay && (
            <div className="overlay-card">
              <h3>ML Validation Overlay</h3>
              <div className="overlay-grid">
                <div>
                  <span>Prediction</span>
                  <strong>{result.ml_overlay.prediction}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{result.ml_overlay.confidence}%</strong>
                </div>
                <div>
                  <span>Signal Strength</span>
                  <strong>{result.ml_overlay.signal_strength}</strong>
                </div>
                <div>
                  <span>Regime</span>
                  <strong>{result.ml_overlay.market_regime}</strong>
                </div>
              </div>
            </div>
          )}

          <div className="chart-card">
            <h3>Historical Close and 20 Day Moving Average</h3>
            <div className="chart-area">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={result.chart_data || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" hide />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="close" stroke="#4ac26c" dot={false} name="Close" />
                  <Line type="monotone" dataKey="sma_20" stroke="#7da7ff" dot={false} name="SMA 20" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Backtester;

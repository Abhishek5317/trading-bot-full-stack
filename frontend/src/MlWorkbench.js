import React, { useMemo, useState } from 'react';
import './MlWorkbench.css';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

function MlWorkbench({ symbol }) {
  const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const probabilityData = useMemo(() => {
    if (!result) {
      return [];
    }

    return [
      { name: 'Up', value: result.up_probability || 0 },
      { name: 'Down', value: result.down_probability || 0 },
    ];
  }, [result]);

  const featureImportanceData = useMemo(() => {
    if (!result?.feature_importances) {
      return [];
    }

    return Object.entries(result.feature_importances).map(([feature, value]) => ({
      feature,
      value: Number((value * 100).toFixed(2)),
    }));
  }, [result]);

  const metricCards = useMemo(() => {
    if (!result?.metrics) {
      return [];
    }

    return [
      { label: 'Accuracy', value: result.metrics.accuracy },
      { label: 'Precision', value: result.metrics.precision },
      { label: 'Recall', value: result.metrics.recall },
      { label: 'F1 Score', value: result.metrics.f1_score },
      { label: 'ROC AUC', value: result.metrics.roc_auc ?? 'N/A' },
    ];
  }, [result]);

  const runMlAnalysis = () => {
    if (!symbol) {
      return;
    }

    setLoading(true);
    setResult(null);

    fetch(`${API_URL}/api/ml-insights/${symbol}`)
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to run ML analysis.');
        }

        return data;
      })
      .then((data) => setResult(data))
      .catch((error) => setResult({ error: error.message }))
      .finally(() => setLoading(false));
  };

  return (
    <div className="ml-workbench">
      <div className="section-header">
        <div>
          <p className="section-tag">Primary Module</p>
          <h2>Machine Learning Signal Engine</h2>
        </div>
        <button onClick={runMlAnalysis} disabled={!symbol || loading}>
          {loading ? 'Running ML Analysis...' : 'Run ML Analysis'}
        </button>
      </div>

      <p className="section-description">
        This module transforms historical OHLCV data into engineered market features, trains an ensemble of classifiers, validates them using time series splits, and produces an interpretable directional signal.
      </p>

      {loading && <div className="loading-state">Preparing features, training models, and evaluating validation performance...</div>}

      {result?.error && <div className="error-state">{result.error}</div>}

      {result && !result.error && (
        <div className="ml-results">
          <div className="ml-hero-grid">
            <div className="hero-kpi">
              <span>Predicted Signal</span>
              <strong>{result.prediction}</strong>
            </div>
            <div className="hero-kpi">
              <span>Confidence</span>
              <strong>{result.confidence}%</strong>
            </div>
            <div className="hero-kpi">
              <span>Validation Accuracy</span>
              <strong>{result.model_accuracy}%</strong>
            </div>
            <div className="hero-kpi">
              <span>Signal Strength</span>
              <strong>{result.signal_strength}</strong>
            </div>
            <div className="hero-kpi">
              <span>Market Regime</span>
              <strong>{result.market_regime}</strong>
            </div>
            <div className="hero-kpi">
              <span>Risk Level</span>
              <strong>{result.risk_level}</strong>
            </div>
          </div>

          <div className="summary-card">
            <h3>Model Summary</h3>
            <p>{result.summary}</p>
            <div className="summary-meta">
              <span>Symbol: {result.symbol}</span>
              <span>As of: {result.as_of}</span>
              <span>Training Samples: {result.training_samples}</span>
              <span>Validation Samples: {result.validation_samples}</span>
            </div>
            <div className="stack-list">
              {result.model_stack?.map((modelName) => (
                <span key={modelName}>{modelName}</span>
              ))}
            </div>
          </div>

          <div className="dual-chart-grid">
            <div className="chart-card">
              <h3>Price Trend and 20 Day Moving Average</h3>
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

            <div className="chart-card">
              <h3>Directional Probability</h3>
              <div className="chart-area">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={probabilityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" name="Probability %">
                      {probabilityData.map((entry) => (
                        <Cell key={entry.name} fill={entry.name === 'Up' ? '#4ac26c' : '#ff7b7b'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="metrics-grid">
            {metricCards.map((metric) => (
              <div className="metric-card" key={metric.label}>
                <span>{metric.label}</span>
                <strong>{typeof metric.value === 'number' ? `${metric.value}%` : metric.value}</strong>
              </div>
            ))}
          </div>

          <div className="analysis-grid">
            <div className="chart-card">
              <h3>Top Feature Importances</h3>
              <div className="chart-area tall-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={featureImportanceData} layout="vertical" margin={{ left: 25, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 'auto']} />
                    <YAxis type="category" dataKey="feature" width={120} />
                    <Tooltip />
                    <Bar dataKey="value" name="Importance %" fill="#4ac26c" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="snapshot-card">
              <h3>Latest Feature Snapshot</h3>
              <div className="snapshot-grid">
                {Object.entries(result.latest_feature_snapshot || {}).map(([key, value]) => (
                  <div className="snapshot-item" key={key}>
                    <span>{key}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="fold-card">
            <h3>Time Series Validation Folds</h3>
            <div className="fold-table">
              <div className="fold-row fold-head">
                <span>Fold</span>
                <span>Train Samples</span>
                <span>Validation Samples</span>
                <span>Accuracy</span>
              </div>
              {(result.fold_details || []).map((fold) => (
                <div className="fold-row" key={fold.fold}>
                  <span>{fold.fold}</span>
                  <span>{fold.train_samples}</span>
                  <span>{fold.validation_samples}</span>
                  <span>{fold.accuracy}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default MlWorkbench;

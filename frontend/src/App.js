import React, { useEffect, useState } from 'react';
import './App.css';
import AccountInfo from './AccountInfo';
import MlWorkbench from './MlWorkbench';
import Backtester from './Backtester';

function App() {
  const [accountInfo, setAccountInfo] = useState(null);
  const [symbol, setSymbol] = useState('AAPL');
  const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

  useEffect(() => {
    fetch(`${API_URL}/api/account`)
      .then((response) => response.json())
      .then((data) => setAccountInfo(data))
      .catch(() => setAccountInfo({ error: 'Could not connect to backend' }));
  }, [API_URL]);

  return (
    <div className="app-shell">
      <header className="hero-section">
        <div className="hero-copy">
          <p className="hero-eyebrow">Final Minor Project</p>
          <h1>Machine Learning Driven Trading Intelligence Platform</h1>
          <p className="hero-subtitle">
            A full stack trading bot where the core KPI is machine learning signal quality, confidence, and model validation.
          </p>
        </div>

        <div className="symbol-card">
          <label htmlFor="symbol-input">Analyse a stock symbol</label>
          <input
            id="symbol-input"
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase().trim())}
            placeholder="AAPL, MSFT, NVDA, TSLA"
            maxLength={10}
          />
          <p className="symbol-helper">
            The ML module performs feature engineering, ensemble classification, time series validation, and live signal interpretation.
          </p>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="dashboard-panel wide-panel">
          <AccountInfo accountData={accountInfo} />
        </section>

        <section className="dashboard-panel wide-panel">
          <MlWorkbench symbol={symbol} />
        </section>

        <section className="dashboard-panel wide-panel">
          <Backtester symbol={symbol} />
        </section>
      </main>
    </div>
  );
}

export default App;

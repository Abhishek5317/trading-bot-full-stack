import React from 'react';
import './AccountInfo.css';

function AccountInfo({ accountData }) {
  if (!accountData) {
    return <div className="account-info-container">Loading account data...</div>;
  }

  if (accountData.error) {
    return <div className="account-info-container">{accountData.error}</div>;
  }

  const cards = [
    { label: 'Status', value: accountData.status },
    { label: 'Account Number', value: accountData.account_number },
    { label: 'Portfolio Value', value: `$${accountData.portfolio_value}` },
    { label: 'Buying Power', value: `$${accountData.buying_power}` },
    { label: 'Cash', value: `$${accountData.cash}` },
  ];

  return (
    <div className="account-info-container">
      <div className="section-header">
        <div>
          <p className="section-tag">System Context</p>
          <h2>Broker Account Overview</h2>
        </div>
      </div>

      <p className="section-description">
        This section provides live account context for the trading workflow and supports the broader ML driven decision dashboard.
      </p>

      <div className="account-info-grid">
        {cards.map((card) => (
          <div className="account-info-card" key={card.label}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default AccountInfo;

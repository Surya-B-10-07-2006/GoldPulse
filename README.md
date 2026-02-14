# 📈 GoldPulse

Streamlit app for predicting precious metal prices using ARIMA forecasting with real-time data and multi-currency support.

## Features
- ARIMA time series forecasting (1-day & 7-day predictions)
- Multi-metal support: Gold, Silver, Platinum
- 10 currency support with proper symbols
- Flexible units: Ounce, Gram, Kilogram, Sovereign
- Animated neon-style visualizations
- Historical data export (CSV)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m streamlit run app.py
```

1. Select metal (Gold/Silver/Platinum)
2. Choose currency and unit
3. Set date range (minimum 2 weeks)
4. Click "Run Prediction"

## Technical Stack

- **Model**: ARIMA (adaptive order selection)
- **Data Source**: Yahoo Finance API
- **Dependencies**: streamlit, yfinance, pandas, numpy, matplotlib, statsmodels, scipy

## Requirements

- Minimum 10 data points (2 weeks recommended)
- Active internet connection
- Static exchange rates (USD, EUR, GBP, INR, JPY, AUD, CAD, CHF, CNY, AED)

---

**GoldPulse** — Precious Metal Price Prediction

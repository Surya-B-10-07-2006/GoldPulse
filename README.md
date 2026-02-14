# 📈 GoldPulse - Precious Metal Price Prediction

A modern, interactive Streamlit web application for predicting precious metal prices using ARIMA time series forecasting with real-time data visualization and multi-currency support.

## ✨ Features

### Core Functionality
- **Real-time Data Fetching**: Live precious metal prices via Yahoo Finance API
- **ARIMA Forecasting**: Statistical time series prediction model
- **7-Day Forecasting**: Weekly price predictions with confidence metrics
- **Multi-Metal Support**: Gold (GC=F), Silver (SI=F), and Platinum (PL=F)
- **Neon-Style Animated Visualizations**: Dynamic price charts with glowing effects and trend indicators
- **Historical Data Analysis**: View and download past 10 days of data

### Currency & Units
- **10 Currency Support**: $ (USD), € (EUR), £ (GBP), ₹ (INR), ¥ (JPY), A$ (AUD), C$ (CAD), CHF, ¥ (CNY), AED
- **Currency Symbol Display**: Proper symbols shown instead of currency codes
- **Flexible Units**: Per Ounce, Per Gram, Per Kilogram, Sovereign (8g for gold)
- **Multi-Currency Conversion Table**: View predictions across all currencies

### User Experience
- **Animated Price Display**: Real-time counting animations for current and predicted prices
- **Interactive Neon Charts**: 30-frame smooth animation with trend arrows and forecast visualization
- **7-Day Forecast**: Weekly outlook with price statistics
- **Price Labels**: Exact price values displayed on each data point
- **Responsive Design**: Modern dark-themed UI with neon glow effects
- **Data Export**: Download historical data as CSV

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd GoldPulse
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Running the Application

```bash
python -m streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Select Metal**: Choose from Gold, Silver, or Platinum
2. **Select Currency**: Pick your preferred currency from 10 options
3. **Select Unit**: Choose measurement unit (ounce, gram, kilogram, or sovereign)
4. **Set Date Range**: Pick start and end dates (minimum 2 weeks of data required)
5. **Run Prediction**: Click the "🚀 Run Prediction" button

### Output
- Current price with real-time counting animation
- ARIMA forecast for next trading day with animated display
- **7-Day Forecast**: Weekly outlook with average, highest, and lowest prices
- Neon-style dynamic price trend chart with 30-frame smooth animation
- Exact price labels on all data points
- Color-coded trend indicators (green for up, red for down)
- Golden dashed forecast line with directional arrows
- Multi-currency conversion table
- Historical data table (last 10 days)
- CSV download option

## 🛠️ Technical Details

### Models
- **ARIMA**: Auto-Regressive Integrated Moving Average
  - Adaptive order selection based on data size
  - Order (2,1,0) for <20 points, (3,1,0) for <50 points, (5,1,0) for 50+ points
  - 7-day forecasting for weekly predictions

### Data Source
- **Yahoo Finance API** via yfinance library
- Real-time OHLCV (Open, High, Low, Close, Volume) data

### Dependencies
- `streamlit` - Web application framework
- `yfinance` - Financial data fetching
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `statsmodels` - ARIMA modeling
- `scipy` - Statistical computations

## ⚙️ Configuration

### Supported Tickers
- **GC=F**: Gold Futures
- **SI=F**: Silver Futures
- **PL=F**: Platinum Futures

### Exchange Rates (Built-in)
- USD: 1.00 (base)
- EUR: 0.92
- GBP: 0.80
- INR: 82.00
- JPY: 148.00
- AUD: 1.52
- CAD: 1.36
- CHF: 0.91
- CNY: 7.20
- AED: 3.67

### Data Requirements
- Minimum 10 data points for ARIMA training
- Recommended: At least 2 weeks of historical data
- Maximum: Any historical range supported by Yahoo Finance

## 🔧 Error Handling

✅ **Robust Error Management**
- Invalid date range validation
- Future date correction (auto-adjusts to current date)
- Insufficient data detection
- API failure handling
- Model training error recovery
- Comprehensive logging

## 🎨 Features Highlights

### Animated Price Display
- Real-time counting animation from 0 to actual price
- Smooth 30-step transitions for both current and predicted prices
- Color-coded displays (blue for current, orange for prediction)
- Currency symbols displayed (e.g., $2,045.32 instead of USD 2045.32)
- Centered layout with label above and price value below

### Neon-Style Chart Visualization
- **30-Frame Smooth Animation**: Progressive rendering with fluid motion
- **Glowing Effects**: Multi-layer neon glow on lines and markers
- **Dark Theme**: Professional dark background (#0a0e27) with vibrant colors
- **Color-Coded Segments**: 
  - 🟢 Neon green (#00ff41) for price increases
  - 🔴 Neon red (#ff0055) for price decreases
  - 🔵 Neon cyan (#00d4ff) for neutral movements
  - 🟡 Golden (#ffd700) for forecast projection

### Trend Indicators
- **Directional Arrows**: ↑ for uptrends, ↓ for downtrends
- **Glowing Arrow Effects**: Multi-layer glow for visual emphasis
- **Smart Display**: Arrows shown only for significant price changes

### Price Labels
- Exact price values displayed on every data point
- Color-matched rounded boxes for readability
- Forecast price clearly labeled on prediction point

### Forecast Visualization
- Dashed golden line projection from last price to forecast
- Animated drawing in final 5 frames
- Directional arrow showing forecast trend
- Clear "Forecast" label on x-axis

### Professional Polish
- Responsive wide layout with sidebar controls
- Neon grid overlay for easy reading
- Consistent y-axis scaling for smooth viewing
- Date labels with 45° rotation for clarity

### 7-Day Forecasting
- **Weekly Predictions**: Next 7 days price forecast
- **Statistical Metrics**: Average, highest, lowest prices
- **Neon Visualization**: Golden glowing chart with price labels
- **Date Labels**: Clear daily breakdown

## 📊 Output Examples

### Price Display
- **Current Price**: Real-time counting animation with proper currency symbol (e.g., $, €, £, ₹)
- **ARIMA Prediction**: Next-day forecast with animated number display and currency symbol
- **7-Day Statistics**: Average, highest, and lowest prices with currency symbols

### Neon Chart Features
- **Last 20 Data Points**: Progressive visualization with smooth animation
- **Exact Price Labels**: Value displayed on each marker point
- **Trend Arrows**: Directional indicators (↑/↓) between consecutive points
- **Color-Coded Lines**: Green for increases, red for decreases
- **Glowing Effects**: Multi-layer neon glow on all elements
- **Forecast Extension**: Golden dashed line with directional arrow
- **Dark Theme**: Professional neon-style dark background
- **Neon Grid**: Cyan glowing grid overlay
- **Date Labels**: Clear x-axis labels including "Forecast" marker

## 🐛 Known Limitations

- Exchange rates are static (not real-time)
- Requires active internet connection for data fetching
- Limited to Yahoo Finance data availability
- Forecast accuracy depends on historical data quality and market conditions

## 📝 License

Copyright © 2026 GoldPulse

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**GoldPulse** — Predicting Tomorrow's Gold Price Today

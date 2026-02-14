import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import logging
import warnings
import time
warnings.filterwarnings('ignore')

np.random.seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data available for {ticker} between {start_date} and {end_date}. Please check your date range.")
        if 'Close' not in data.columns:
            raise ValueError("Close column missing")
        return data[['Close', 'Open', 'High', 'Low', 'Volume']]
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

def train_arima(series, steps=1):
    try:
        if len(series) < 10:
            raise ValueError(f"Need at least 10 data points for ARIMA, got {len(series)}. Please select a longer date range.")
        
        # Adjust ARIMA order based on data size
        if len(series) < 20:
            order = (2, 1, 0)
        elif len(series) < 50:
            order = (3, 1, 0)
        else:
            order = (5, 1, 0)
        
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        
        if steps == 1:
            return float(forecast.values[0])
        else:
            return forecast.values
    except Exception as e:
        logger.error(f"ARIMA error: {e}")
        raise

st.set_page_config(page_title="GoldPulse", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main-header {text-align: center; color: #FFD700; font-size: 3em; font-weight: bold; animation: fadeIn 1s;}
    .sub-header {text-align: center; color: #888; font-size: 1.2em; margin-bottom: 30px;}
    .footer {text-align: center; color: #888; margin-top: 50px; padding: 20px;}
    .gold-coin {text-align: center; font-size: 5em; animation: spin 2s linear infinite;}
    @keyframes spin {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(360deg); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes pulseGreen {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    @keyframes pulseRed {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .price-display {
        text-align: center;
        padding: 20px;
    }
    .price-label {
        color: #666;
        font-size: 1.2em;
        margin-bottom: 10px;
    }
    .price-value {
        color: #1976d2;
        font-size: 2.5em;
        font-weight: bold;
    }
    .price-increase {
        animation: pulseGreen 0.5s ease-in-out;
        color: #00ff41 !important;
    }
    .price-decrease {
        animation: pulseRed 0.5s ease-in-out;
        color: #ff3838 !important;
    }
    .trend-arrow {
        font-size: 1.5em;
        margin-left: 10px;
    }
    .chart-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #f5f9ff 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s;
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #00ff41;
        border-radius: 50%;
        animation: pulseGreen 1s infinite;
        margin-right: 8px;
    }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 GoldPulse</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Accurate gold price predictions at your fingertips</p>', unsafe_allow_html=True)

exchange_rates = {"USD": 1, "EUR": 0.92, "GBP": 0.80, "INR": 82, "JPY": 148, "AUD": 1.52, "CAD": 1.36, "CHF": 0.91, "CNY": 7.2, "AED": 3.67}
currency_names = {"USD": "US Dollar", "EUR": "Euro", "GBP": "British Pound", "INR": "Indian Rupee", "JPY": "Japanese Yen", "AUD": "Australian Dollar", "CAD": "Canadian Dollar", "CHF": "Swiss Franc", "CNY": "Chinese Yuan", "AED": "UAE Dirham"}
metal_names = {"GC=F": "Gold", "SI=F": "Silver", "PL=F": "Platinum"}
metal_icons = {"GC=F": "🪙", "SI=F": "💍", "PL=F": "💎"}

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.selectbox("🏆 Select Metal", list(metal_names.keys()), format_func=lambda x: metal_names[x])
    currency = st.selectbox("💱 Select Currency", list(exchange_rates.keys()))
    
    if ticker == "GC=F":
        unit = st.radio("📏 Select Unit", ["Per Ounce", "Per Gram", "Sovereign (8g)"])
    else:
        unit = st.radio("📏 Select Unit", ["Per Ounce", "Per Gram", "Per Kilogram"])
    
    st.divider()
    start_date = st.date_input("📅 Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("📅 End Date", value=datetime.now())
    st.divider()
    predict_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

if predict_btn:
    try:
        # Validate date range
        if start_date >= end_date:
            st.error("❌ Invalid date range: Start date must be before end date.")
            st.stop()
        
        if end_date > datetime.now().date():
            st.warning("⚠️ End date is in the future. Using today's date instead.")
            end_date = datetime.now().date()
        
        rate = exchange_rates[currency]
        if unit == "Per Gram":
            unit_divisor = 31.1035
            unit_label = "gram"
        elif unit == "Sovereign (8g)":
            unit_divisor = 31.1035 / 8
            unit_label = "sovereign (8g)"
        elif unit == "Per Kilogram":
            unit_divisor = 31.1035 / 1000
            unit_label = "kilogram"
        else:
            unit_divisor = 1
            unit_label = "ounce"
        
        st.markdown(f'<div class="gold-coin">{metal_icons[ticker]}</div>', unsafe_allow_html=True)
        
        with st.spinner("🔄 Fetching data..."):
            df = fetch_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if len(df) < 10:
            st.error(f"❌ Insufficient data: Only {len(df)} records found. Please select a date range with at least 2 weeks of trading data.")
            st.stop()
        
        st.success(f"✅ Fetched {len(df)} records")
        
        current_price = float(df['Close'].values[-1]) * rate / unit_divisor
        
        with st.spinner("🔄 Running ARIMA Forecasting..."):
            # Generate forecasts for 1 and 7 days
            forecast_7_raw = train_arima(df['Close'], steps=7)
            # Convert to numpy array and apply conversions
            forecast_7_array = np.array(forecast_7_raw, dtype=float)
            forecast_prices = forecast_7_array * rate / unit_divisor
            arima_price = float(forecast_prices[0])  # Next day
            forecast_7day = forecast_prices
        
        # Currency symbols mapping
        currency_symbols = {
            "USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", 
            "JPY": "¥", "AUD": "A$", "CAD": "C$", 
            "CHF": "CHF", "CNY": "¥", "AED": "AED"
        }
        currency_symbol = currency_symbols.get(currency, currency)
        
        col1, col2 = st.columns(2)
        
        # Determine price trend direction
        prev_price = float(df['Close'].values[-2]) * rate / unit_divisor if len(df) > 1 else current_price
        price_trend = "increase" if current_price > prev_price else "decrease"
        trend_icon = "↑" if price_trend == "increase" else "↓"
        trend_class = "price-increase" if price_trend == "increase" else "price-decrease"
        
        # Animated price displays
        with col1:
            st.markdown("<h3 style='text-align: center; color: #666;'>💰 Current Price</h3>", unsafe_allow_html=True)
            price_placeholder1 = st.empty()
            steps = 30
            for i in range(steps + 1):
                animated_value = (current_price / steps) * i
                price_placeholder1.markdown(
                    f"<h1 style='text-align: center; color: #1976d2; margin-top: 10px;'>{currency_symbol}{animated_value:,.2f}</h1>", 
                    unsafe_allow_html=True
                )
                time.sleep(0.02)
        
        with col2:
            st.markdown("<h3 style='text-align: center; color: #666;'>📈 ARIMA Prediction</h3>", unsafe_allow_html=True)
            price_placeholder2 = st.empty()
            for i in range(steps + 1):
                animated_value = (arima_price / steps) * i
                price_placeholder2.markdown(
                    f"<h1 style='text-align: center; color: #ff6f00; margin-top: 10px;'>{currency_symbol}{animated_value:,.2f}</h1>", 
                    unsafe_allow_html=True
                )
                time.sleep(0.02)
        
        st.markdown("---")
        st.markdown("### 📈 Live Price Trend & Forecast")
        
        # Prepare data for matplotlib animation
        sample_df = df.tail(20).copy()
        sample_df['Price'] = sample_df['Close'] * rate / unit_divisor
        dates = [d.strftime('%m/%d') for d in sample_df.index]
        prices = sample_df['Price'].values
        
        # Create 30-frame smooth animation
        chart_placeholder = st.empty()
        total_frames = 30
        
        for frame in range(total_frames + 1):
            fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0a0e27')
            ax.set_facecolor('#0a0e27')
            
            # Calculate how many points to show based on frame
            progress = frame / total_frames
            points_to_show = int(progress * len(prices))
            
            if points_to_show > 0:
                # Draw all line segments up to current progress
                for j in range(1, points_to_show):
                    # Determine trend color
                    if prices[j] > prices[j-1]:
                        color = '#00ff41'  # Green up
                        arrow = '↑'
                    elif prices[j] < prices[j-1]:
                        color = '#ff0055'  # Red down
                        arrow = '↓'
                    else:
                        color = '#00d4ff'  # Cyan neutral
                        arrow = ''
                    
                    # Glow effect layers
                    for glow_width in [12, 8, 4]:
                        ax.plot([j-1, j], [prices[j-1], prices[j]], 
                               color=color, linewidth=glow_width, alpha=0.15, solid_capstyle='round')
                    
                    # Main bright line
                    ax.plot([j-1, j], [prices[j-1], prices[j]], 
                           color=color, linewidth=3, alpha=1, solid_capstyle='round', zorder=5)
                    
                    # Add trend arrow
                    if arrow and abs(prices[j] - prices[j-1]) > 0.01:
                        mid_x = j - 0.5
                        mid_y = (prices[j] + prices[j-1]) / 2
                        # Glow arrow
                        ax.text(mid_x, mid_y, arrow, fontsize=24, color=color, 
                               ha='center', va='center', weight='bold', alpha=0.3, zorder=2)
                        # Bright arrow
                        ax.text(mid_x, mid_y, arrow, fontsize=18, color=color, 
                               ha='center', va='center', weight='bold', alpha=1, zorder=6)
                
                # Draw markers
                for j in range(points_to_show):
                    if j == 0:
                        marker_color = '#00d4ff'
                    elif prices[j] > prices[j-1]:
                        marker_color = '#00ff41'
                    elif prices[j] < prices[j-1]:
                        marker_color = '#ff0055'
                    else:
                        marker_color = '#00d4ff'
                    
                    # Glowing markers
                    ax.plot(j, prices[j], marker='o', markersize=18, color=marker_color, alpha=0.2, zorder=3)
                    ax.plot(j, prices[j], marker='o', markersize=11, color=marker_color, alpha=0.5, zorder=4)
                    ax.plot(j, prices[j], marker='o', markersize=7, color=marker_color, alpha=1, zorder=5)
                    
                    # Add price label on each marker
                    ax.text(j, prices[j], f'{prices[j]:.2f}', 
                           fontsize=8, ha='center', va='bottom', color='white', 
                           weight='bold', zorder=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=marker_color, 
                                    edgecolor='white', linewidth=0.5, alpha=0.8))
            
            # Add forecast visualization on final frames
            if frame >= total_frames - 5 and points_to_show >= len(prices):
                forecast_progress = (frame - (total_frames - 5)) / 5
                
                # Forecast line with glow
                for glow_width in [12, 8, 4]:
                    ax.plot([len(prices)-1, len(prices)-1 + forecast_progress], 
                           [prices[-1], prices[-1] + (arima_price - prices[-1]) * forecast_progress], 
                           linestyle='--', color='#ffd700', linewidth=glow_width, alpha=0.2)
                
                # Main forecast line
                ax.plot([len(prices)-1, len(prices)-1 + forecast_progress], 
                       [prices[-1], prices[-1] + (arima_price - prices[-1]) * forecast_progress], 
                       linestyle='--', color='#ffd700', linewidth=3, alpha=1, zorder=5)
                
                # Forecast marker
                if forecast_progress >= 0.9:
                    ax.plot(len(prices), arima_price, marker='o', markersize=18, 
                           color='#ffd700', alpha=0.3, zorder=3)
                    ax.plot(len(prices), arima_price, marker='o', markersize=12, 
                           color='#ffd700', alpha=1, zorder=5)
                    
                    # Add forecast price label
                    ax.text(len(prices), arima_price, f'{arima_price:.2f}', 
                           fontsize=9, ha='center', va='bottom', color='white', 
                           weight='bold', zorder=8,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffd700', 
                                    edgecolor='white', linewidth=0.5, alpha=0.9))
                    
                    # Forecast arrow
                    forecast_change = arima_price - prices[-1]
                    if abs(forecast_change) > 0.01:
                        arrow_symbol = '↑' if forecast_change > 0 else '↓'
                        mid_forecast = (prices[-1] + arima_price) / 2
                        
                        # Glow arrow
                        ax.text(len(prices) - 0.5, mid_forecast, arrow_symbol, 
                               fontsize=28, color='#ffd700', ha='center', va='center', 
                               weight='bold', alpha=0.3, zorder=2)
                        # Bright arrow
                        ax.text(len(prices) - 0.5, mid_forecast, arrow_symbol, 
                               fontsize=22, color='#ffd700', ha='center', va='center', 
                               weight='bold', alpha=1, zorder=6)
            
            # Set x-axis
            all_labels = dates + ['Forecast']
            ax.set_xticks(range(len(all_labels)))
            ax.set_xticklabels(all_labels, rotation=45, ha='right')
            
            # Styling
            ax.grid(True, alpha=0.15, linestyle='--', color='#00d4ff', linewidth=0.5)
            ax.set_xlabel('Date', fontsize=13, weight='bold', color='#00d4ff')
            ax.set_ylabel(f'Price ({currency})', fontsize=13, weight='bold', color='#00d4ff')
            ax.set_title('🌟 LIVE PRICE TREND & FORECAST 🌟', fontsize=16, 
                        fontweight='bold', color='#ffd700', pad=20)
            ax.tick_params(colors='#00d4ff', labelsize=10)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#00d4ff')
                spine.set_linewidth(2)
            
            # Set consistent y-axis limits
            y_min = min(prices.min(), arima_price) * 0.995
            y_max = max(prices.max(), arima_price) * 1.005
            ax.set_ylim(y_min, y_max)
            
            plt.tight_layout()
            chart_placeholder.pyplot(fig)
            plt.close()
            time.sleep(0.05)
        
        time.sleep(0.3)
        
        st.markdown("---")
        st.markdown("### 📅 7-Day Forecast")
        
        # Create 7-day forecast chart
        fig7, ax7 = plt.subplots(figsize=(12, 6), facecolor='#0a0e27')
        ax7.set_facecolor('#0a0e27')
        
        forecast_dates_7 = [(datetime.now() + timedelta(days=i+1)).strftime('%m/%d') for i in range(7)]
        
        # Plot with glow effect
        for glow in [8, 4]:
            ax7.plot(range(7), forecast_7day, color='#ffd700', linewidth=glow, alpha=0.15)
        ax7.plot(range(7), forecast_7day, color='#ffd700', linewidth=3, marker='o', 
                markersize=10, markerfacecolor='#ffd700', label='7-Day Forecast')
        
        # Add price labels
        for i, price in enumerate(forecast_7day):
            ax7.text(i, price, f'{price:.2f}', fontsize=9, ha='center', va='bottom', 
                    color='white', weight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffd700', alpha=0.8))
        
        ax7.set_xticks(range(7))
        ax7.set_xticklabels(forecast_dates_7, rotation=45, ha='right')
        ax7.grid(True, alpha=0.15, linestyle='--', color='#00d4ff')
        ax7.set_xlabel('Date', fontsize=12, weight='bold', color='#00d4ff')
        ax7.set_ylabel(f'Price ({currency})', fontsize=12, weight='bold', color='#00d4ff')
        ax7.set_title('📈 7-Day Price Forecast', fontsize=14, weight='bold', color='#ffd700', pad=15)
        ax7.tick_params(colors='#00d4ff')
        for spine in ax7.spines.values():
            spine.set_edgecolor('#00d4ff')
            spine.set_linewidth(2)
        plt.tight_layout()
        st.pyplot(fig7)
        
        # Display 7-day statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Price", f"{currency_symbol}{forecast_7day.mean():,.2f}")
        with col2:
            st.metric("Highest", f"{currency_symbol}{forecast_7day.max():,.2f}")
        with col3:
            st.metric("Lowest", f"{currency_symbol}{forecast_7day.min():,.2f}")
        
        st.markdown("---")
        st.markdown("### 🌍 Multi-Currency Conversion")
        
        multi_curr = []
        for curr, r in exchange_rates.items():
            row = {
                "Currency": f"{curr} - {currency_names[curr]}", 
                "Current": f"{curr} {(current_price / rate * r):,.2f}",
                "ARIMA": f"{curr} {(arima_price / rate * r):,.2f}"
            }
            multi_curr.append(row)
        
        st.dataframe(pd.DataFrame(multi_curr), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 📋 Historical Data (Last 10 Days)")
        
        hist_df = df[['Close']].tail(10).copy()
        hist_df['Close'] = hist_df['Close'] * rate / unit_divisor
        hist_df.index = hist_df.index.strftime('%Y-%m-%d')
        hist_df = hist_df.rename(columns={'Close': f'Price ({currency})'})
        st.dataframe(hist_df, use_container_width=True)
        
        csv = hist_df.to_csv().encode('utf-8')
        st.download_button("📥 Download CSV", csv, "goldpulse_data.csv", "text/csv")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.markdown('<div class="footer">GoldPulse — Predicting Tomorrow\'s Gold Price Today | © 2026</div>', unsafe_allow_html=True)

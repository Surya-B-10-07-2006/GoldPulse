import matplotlib
matplotlib.use('Agg')
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
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
    .main-header {text-align: center; color: #FFD700; font-size: 3em; font-weight: bold;}
    .sub-header {text-align: center; color: #888; font-size: 1.2em; margin-bottom: 30px;}
    .footer {text-align: center; color: #888; margin-top: 50px; padding: 20px;}
    </style>
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
        if start_date >= end_date:
            st.error("❌ Invalid date range: Start date must be before end date.")
            st.stop()
        
        if end_date > datetime.now().date():
            st.warning("⚠️ End date is in the future. Using today's date instead.")
            end_date = datetime.now().date()
        
        rate = exchange_rates[currency]
        if unit == "Per Gram":
            unit_divisor = 31.1035
        elif unit == "Sovereign (8g)":
            unit_divisor = 31.1035 / 8
        elif unit == "Per Kilogram":
            unit_divisor = 31.1035 / 1000
        else:
            unit_divisor = 1
        
        st.markdown(f'<div style="text-align: center; font-size: 5em;">{metal_icons[ticker]}</div>', unsafe_allow_html=True)
        
        with st.spinner("🔄 Fetching data..."):
            df = fetch_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if len(df) < 10:
            st.error(f"❌ Insufficient data: Only {len(df)} records found. Please select a date range with at least 2 weeks of trading data.")
            st.stop()
        
        st.success(f"✅ Fetched {len(df)} records")
        
        current_price = float(df['Close'].values[-1]) * rate / unit_divisor
        
        with st.spinner("🔄 Running ARIMA Forecasting..."): 
            forecast_7_raw = train_arima(df['Close'], steps=7)
            if isinstance(forecast_7_raw, (int, float)):
                forecast_7_array = np.array([forecast_7_raw])
            else:
                forecast_7_array = np.array(forecast_7_raw).flatten()
            forecast_prices = forecast_7_array * rate / unit_divisor
            arima_price = float(forecast_prices[0])
            forecast_7day = forecast_prices
        
        currency_symbols = {
            "USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", 
            "JPY": "¥", "AUD": "A$", "CAD": "C$", 
            "CHF": "CHF", "CNY": "¥", "AED": "AED"
        }
        currency_symbol = currency_symbols.get(currency, currency)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='text-align: center; color: #666;'>💰 Current Price</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<h1 style='text-align: center; color: #1976d2; margin-top: 10px;'>{currency_symbol}{current_price:,.2f}</h1>", 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown("<h3 style='text-align: center; color: #666;'>📈 ARIMA Prediction</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<h1 style='text-align: center; color: #ff6f00; margin-top: 10px;'>{currency_symbol}{arima_price:,.2f}</h1>", 
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.markdown("### 📈 Price Trend & Forecast")
        
        sample_df = df.tail(20).copy()
        sample_df['Price'] = sample_df['Close'] * rate / unit_divisor
        dates = [d.strftime('%m/%d') for d in sample_df.index]
        prices = sample_df['Price'].values
        
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0a0e27')
        ax.set_facecolor('#0a0e27')
        
        for j in range(1, len(prices)):
            if prices[j] > prices[j-1]:
                color = '#00ff41'
            elif prices[j] < prices[j-1]:
                color = '#ff0055'
            else:
                color = '#00d4ff'
            
            ax.plot([j-1, j], [prices[j-1], prices[j]], color=color, linewidth=3, solid_capstyle='round', zorder=5)
        
        for j in range(len(prices)):
            if j == 0:
                marker_color = '#00d4ff'
            elif prices[j] > prices[j-1]:
                marker_color = '#00ff41'
            elif prices[j] < prices[j-1]:
                marker_color = '#ff0055'
            else:
                marker_color = '#00d4ff'
            
            ax.plot(j, prices[j], marker='o', markersize=7, color=marker_color, zorder=5)
        
        ax.plot([len(prices)-1, len(prices)], [prices[-1], arima_price], 
               linestyle='--', color='#ffd700', linewidth=3, zorder=5)
        ax.plot(len(prices), arima_price, marker='o', markersize=10, color='#ffd700', zorder=5)
        
        all_labels = dates + ['Forecast']
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        
        ax.grid(True, alpha=0.15, linestyle='--', color='#00d4ff', linewidth=0.5)
        ax.set_xlabel('Date', fontsize=13, weight='bold', color='#00d4ff')
        ax.set_ylabel(f'Price ({currency})', fontsize=13, weight='bold', color='#00d4ff')
        ax.set_title('🌟 PRICE TREND & FORECAST 🌟', fontsize=16, fontweight='bold', color='#ffd700', pad=20)
        ax.tick_params(colors='#00d4ff', labelsize=10)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#00d4ff')
            spine.set_linewidth(2)
        
        y_min = min(prices.min(), arima_price) * 0.995
        y_max = max(prices.max(), arima_price) * 1.005
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        st.markdown("### 📅 7-Day Forecast")
        
        fig7, ax7 = plt.subplots(figsize=(12, 6), facecolor='#0a0e27')
        ax7.set_facecolor('#0a0e27')
        
        forecast_dates_7 = [(datetime.now() + timedelta(days=i+1)).strftime('%m/%d') for i in range(7)]
        
        ax7.plot(range(7), forecast_7day, color='#ffd700', linewidth=3, marker='o', 
                markersize=10, markerfacecolor='#ffd700', label='7-Day Forecast')
        
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

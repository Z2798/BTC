import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="BTC Price Predictor", layout="wide")
st.title("📈 Real-Time Bitcoin Price Predictor (5-Min Ahead)")

# Function to fetch 1-minute BTCUSDT candles from Binance
def fetch_binance_ohlcv():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": 120  # last 2 hours
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['close']].astype(float)
    return df

# Simple linear prediction model
def predict_next_5min(prices):
    prices = prices[-60:]  # last 60 minutes
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    future_index = np.array([[len(prices) + 5]])
    prediction = model.predict(future_index)[0][0]
    return prediction

# App logic
try:
    df = fetch_binance_ohlcv()

    st.subheader("Live BTC Price")
    st.metric("Current Price", f"${df['close'].iloc[-1]:,.2f}")
    st.line_chart(df['close'])

    predicted_price = predict_next_5min(df['close'])

    st.subheader("Predicted BTC Price (5 minutes ahead)")
    st.success(f"${predicted_price:,.2f}")

    fig, ax = plt.subplots()
    ax.plot(df['close'][-60:], label="Last 60 min")
    ax.axhline(predicted_price, color="red", linestyle="--", label="5-min Forecast")
    ax.set_title("BTC Price Prediction")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")

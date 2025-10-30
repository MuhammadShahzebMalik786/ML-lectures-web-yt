import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(page_title="🚀 AI Stock Predictor", page_icon="📈")

# Title
st.title("🚀 AI Stock Price Predictor")
st.markdown("### Predict stock prices using Machine Learning!")

# Sidebar
st.sidebar.header("📊 Settings")
stock_symbol = st.sidebar.selectbox(
    "Choose Stock:",
    ["TSLA", "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
)

period = st.sidebar.selectbox(
    "Time Period:",
    ["1mo", "3mo", "6mo", "1y", "2y"]
)

# Functions
@st.cache_data
def get_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data

def prepare_features(data):
    df = data.copy()
    df['Day'] = range(len(df))
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Volume_MA'] = df['Volume'].rolling(5).mean()
    df = df.dropna()
    
    features = ['Day', 'Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'Volume_MA']
    X = df[features]
    y = df['Close']
    
    return X, y, df

# Main app
if st.button("🔮 Predict Stock Price"):
    with st.spinner("Loading data..."):
        # Get data
        data = get_stock_data(stock_symbol, period)
        
        if len(data) > 20:
            # Prepare features
            X, y, processed_data = prepare_features(data)
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make prediction
            latest_features = X.iloc[-1:].values
            prediction = model.predict(latest_features)[0]
            current_price = data['Close'].iloc[-1]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                st.metric("AI Prediction", f"${prediction:.2f}")
            
            with col3:
                change = prediction - current_price
                st.metric("Predicted Change", f"${change:.2f}", f"{change:.2f}")
            
            # Chart
            st.subheader("📈 Price History")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, data['Close'], label='Actual Price', color='blue')
            ax.axhline(y=prediction, color='red', linestyle='--', label=f'AI Prediction: ${prediction:.2f}')
            ax.set_title(f'{stock_symbol} Stock Price')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            st.pyplot(fig)
            
            # Model info
            st.subheader("🤖 Model Performance")
            score = model.score(X, y)
            st.write(f"**Model Accuracy:** {score*100:.1f}%")
            
            # Feature importance
            feature_names = ['Day', 'Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_10', 'Volume_MA']
            importance = abs(model.coef_)
            
            st.subheader("📊 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature'))
            
        else:
            st.error("Not enough data for this period. Try a longer time frame.")

# Info section
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ How it works:")
st.sidebar.markdown("""
1. **Fetches** real stock data
2. **Analyzes** price patterns
3. **Predicts** next price using AI
4. **Shows** confidence level
""")

st.sidebar.markdown("### ⚠️ Disclaimer:")
st.sidebar.markdown("This is for educational purposes only. Not financial advice!")

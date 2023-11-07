import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf  # For fetching data from Yahoo Finance
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Function to preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to make predictions
def make_predictions(model, data, scaler):
    predictions = model.predict(data)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Streamlit app
def main():
    st.title("Stock Price Prediction App")
    
    # Load the pre-trained model
    model = load_model("model.h5")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select a Page", ["Analysis", "Prediction"])
    
    if page == "Analysis":
        st.header("Stock Price Analysis")
        
        # User input for stock symbol and period
        st.subheader("Select Stock and Period")
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
        period_days = st.slider("Select Historical Data Period (in days):", 1, 365, 30)
        
        if symbol:
            # Fetch historical stock price data from Yahoo Finance
            df = yf.download(symbol, period=f"{period_days}d")
            
            st.subheader("Historical Stock Data")
            st.write(df.head())
            
            # Additional analysis plots
            st.subheader("Additional Analysis")
            
            # Moving Average (MA) plot
            ma_period = st.slider("Select MA Period (in days):", 1, 30, 7)
            df['MA'] = df['Close'].rolling(window=ma_period).mean()
            
            # Plot both data and MA
            st.line_chart(df[['Close', 'MA']], use_container_width=True)
            
            # Mean and Standard Deviation (STD)
            mean = df['Close'].mean()
            std = df['Close'].std()
            st.write(f"Mean Price: {mean:.2f}")
            st.write(f"Standard Deviation: {std:.2f}")
    
    elif page == "Prediction":
        st.header("Make Stock Price Predictions")
        
        # User input for stock symbol and period
        st.subheader("Select Stock and Period")
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
        period_days = st.slider("Select Historical Data Period (in days):", 1, 365, 30)
        
        if symbol:
            # Fetch historical stock price data from Yahoo Finance
            df = yf.download(symbol, period=f"{period_days}d")
            
            st.subheader("Historical Stock Data")
            st.write(df.head())
            
            # Preprocess data
            data, scaler = preprocess_data(df)
            
            # Make predictions
            predictions = make_predictions(model, data, scaler)
            
            st.subheader("Predicted Stock Prices")
            st.write(predictions)
            
            st.subheader("Predicted Price Chart")
            st.line_chart(predictions)

if __name__ == "__main__":
    main()

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
    
    st.header("Make Stock Price Predictions")
    
    # User input for stock symbol and period
    st.subheader("Select Stock and Period")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
    period = st.slider("Select Historical Data Period (in years):", 1, 10, 5)
    
    if symbol:
        # Fetch historical stock price data from Yahoo Finance
        df = yf.download(symbol, period=f"{period}y")
        
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

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model
import yfinance as yf

# Function to get stock data from Yahoo Finance
def get_stock_data(symbol):
    try:
        start = datetime(2009, 1, 1)
        end = datetime.now()
        df = yf.download(symbol, start=start, end=end)
        return df
    except:
        return None

# Function to display the "Data Visualization" page
def data_visualization():
    st.title("Data Visualization")
    
    # Input for stock symbol
    symbol = st.text_input("Enter a stock symbol (e.g., GOLD):")
    
    if not symbol:
        st.warning("Please enter a symbol to get data.")
    else:
        df = get_stock_data(symbol)

        if df is None or df.empty:
            st.error("Data not available for this symbol.")
        else:
            # Display stock data
            st.subheader(f"Stock data for {symbol}")
            st.dataframe(df)

            # Plot raw data
            st.subheader("Stock Price Chart")
            st.line_chart(df['Close'])

# Function to display the "Analysis and Prediction" page
def analysis_and_prediction():
    st.title("Analysis and Prediction")
    
    # Input for stock symbol
    symbol = st.text_input("Enter a stock symbol (e.g., GOLD):")
    
    if not symbol:
        st.warning("Please enter a symbol to get data.")
    else:
        df = get_stock_data(symbol)

        if df is None or df.empty:
            st.error("Data not available for this symbol.")
        else:
            # Load the machine learning model
            model = load_model('model.h5')

            # Data preprocessing and prediction code here (adapt to your model)

            # Display analysis results, predictions, and charts

# Create navigation in the sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ("Data Visualization", "Analysis and Prediction"))

# Display the selected page based on user's choice
if page == "Data Visualization":
    data_visualization()
elif page == "Analysis and Prediction":
    analysis_and_prediction()

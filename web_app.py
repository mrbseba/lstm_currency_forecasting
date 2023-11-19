import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf  # For fetching data from Yahoo Finance
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Function to preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to make predictions
def make_predictions(model, data, scaler, num_days):
    predictions = []
    for i in range(num_days):
        # Predict the next day's price
        prediction = model.predict(data[-1].reshape(1, -1))
        predictions.append(prediction[0])
        
        # Append the prediction to the data for the next prediction
        data = np.append(data, prediction.reshape(1, -1), axis=0)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
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
            
            # Plot both data and MA using Plotly
            fig = px.line(df, x=df.index, y=['Close', 'MA'], labels={'index': 'Date', 'value': 'Price'})
            st.plotly_chart(fig, use_container_width=True)
            
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
        period_days = st.slider("Select Historical Data Period (in days):", 1, 365, 60)
        
        if symbol:
            # Fetch historical stock price data from Yahoo Finance
            df = yf.download(symbol, period=f"{period_days}d")
            
            # Ensure the index is a datetime index
            df.index = pd.to_datetime(df.index)

            st.subheader("Historical Stock Data")
            st.write(df.head())
            
            # Preprocess data
            data, scaler = preprocess_data(df)
            
            # Number of days to predict
            num_days = 14
            
            # Make predictions
            predictions = make_predictions(model, data, scaler, num_days)
 
            # Create a DataFrame for predictions with a datetime index
            last_date = df.index[-1]
            date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days)
            prediction_df = pd.DataFrame(predictions, columns=['Predicted Price'], index=date_range)
            
            # Combine historical and predicted data
            combined_df = pd.concat([df['Close'].tail(60), prediction_df['Predicted Price']], axis=1)
            combined_df.columns = ['Close', 'Predicted Price']  # Rename the columns for clarity

            # Create the plot
            combined_fig = px.line(combined_df, x=combined_df.index, y=['Close', 'Predicted Price'], labels={'index': 'Date', 'value': 'Price'}, title='Combined Historical Data and Predicted Prices')

            st.subheader("Combined Historical Data and Predicted Prices")
            st.plotly_chart(combined_fig, use_container_width=True)

if __name__ == "__main__":
    main()

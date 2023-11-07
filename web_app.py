import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to get stock data
def get_stock_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

# Function to preprocess and split data
def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    training_data_len = int(np.ceil(len(scaled_data) * .95))
    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    return x_train, y_train, scaler, training_data_len

# Function to create and train the LSTM model
def create_train_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model

# Function to get the stock price for the next 14 days
def get_next_14_days_price(model, data, scaler, training_data_len):
    test_data = data[training_data_len - 60:, :]
    x_test = []
    
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions

# Streamlit app
def main():
    st.title("Stock Price Forecasting App")
    
    # Sidebar
    st.sidebar.header("User Input")
    
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
    start_date = st.sidebar.date_input("Start Date:", datetime(2010, 1, 1))
    end_date = st.sidebar.date_input("End Date:", datetime(2021, 1, 1))
    
    # Data retrieval
    df = get_stock_data(symbol, start_date, end_date)
    
    st.subheader("Stock Data")
    st.write(df.head())
    
    st.subheader("Closing Price Chart")
    st.line_chart(df['Close'])
    
    x_train, y_train, scaler, training_data_len = preprocess_data(df)
    
    st.subheader("Training LSTM Model")
    model = create_train_model(x_train, y_train)
    st.write("Model training complete.")
    
    # Analysis Page
    st.header("Analysis")
    
    st.subheader("Stock Data Summary")
    st.write(df.describe())
    
    st.subheader("Closing Price Statistics")
    st.write(f"Minimum Price: {df['Close'].min()}")
    st.write(f"Maximum Price: {df['Close'].max()}")
    st.write(f"Mean Price: {df['Close'].mean()}")
    st.write(f"Standard Deviation: {df['Close'].std()}")
    
    st.subheader("Closing Price Distribution")
    st.bar_chart(df['Close'])
    
    st.subheader("Closing Price vs. Date")
    st.line_chart(df[['Date', 'Close']].set_index('Date'))
    
    # Prediction Page
    st.header("Prediction")
    
    prediction_days = st.slider("Select Number of Days for Prediction:", min_value=1, max_value=30, value=14)
    next_14_days = df.iloc[-1]['Date'] + timedelta(days=prediction_days)
    
    st.subheader(f"Predicting Stock Prices for the Next {prediction_days} Days")
    
    predictions = get_next_14_days_price(model, df[['Close']].values, scaler, training_data_len)
    
    prediction_dates = [next_14_days + timedelta(days=i) for i in range(prediction_days)]
    prediction_df = pd.DataFrame({'Date': prediction_dates, 'Predicted Price': predictions.flatten()})
    
    st.subheader("Predicted Prices")
    st.write(prediction_df)
    
    st.subheader("Predicted Price Chart")
    st.line_chart(prediction_df.set_index('Date'))

if __name__ == "__main__":
    main()

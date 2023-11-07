import streamlit as st
import numpy as np
import pandas as pd
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
    
    # User input for data
    st.subheader("Input Historical Stock Prices")
    uploaded_file = st.file_uploader("Upload a CSV file with historical stock prices:", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Uploaded Stock Data")
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

# import the libraries
import numpy as np
import pandas as pd

# import the library for the plot
# import matplotlib.pyplot as plt
from plotly import graph_objs as go # for more cute plot

# import library for date
from datetime import datetime

# library for model
from keras.models import load_model

# conect streamlit
import streamlit as st

# import the stock data from yahoo
import yfinance as yf

# method for getting data for stock price
def get_gold_data(symbol):
    try:
        start = datetime(2009, 1, 1)
        end = datetime.now()
        df = yf.download(symbol, start=start, end=end)
        return df
    except:
        return None

# put the title for the page
st.title("Stock Price Forecasting")

# input the stock
symbol = st.text_input("Enter a stock symbol (e.g. GOLD):")

# create conection for yahoo data
if not symbol:
    st.warning("Please enter a symbol to get data.")
else:
    df = get_gold_data(symbol)

    if df is None or df.empty:
        st.error("Data not available for this symbol.")
    else:
        # get info from yahoo
        stock_info = yf.Ticker(symbol).info

        stock_info.keys()  # for other properties you can explore
        company_name = stock_info['shortName']
        st.subheader(company_name)

        # get the price for element
        market_price = stock_info['regularMarketPrice']
        previous_close_price = stock_info['regularMarketPreviousClose']
        st.write('market price: ', market_price,
                 'previous close price: ', previous_close_price)
        # st.write()

        # display the title for stock name
        st.write(f"Stock data for {symbol}")
        st.dataframe(df)

        # fancy chart for the display the open and closed price
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name="Stock Open"))
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Stock Close"))
            fig.update_layout(title_text='Stock Price',
                            xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        # st.subheader('Closing Price vs Time Chart')
        plot_raw_data()

        # fancy code for 100 & 200MA
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100MA', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='200MA', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df.Close, mode='lines', name='Price', line=dict(color='white')))

        fig.update_layout(title='Price vs Time Chart with 100MA & 200MA',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        xaxis_rangeslider_visible=True)

        st.plotly_chart(fig)



        # Splitting data into Training and Testing
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        # print(data_training.shape)
        # print(data_testing.shape)

        # scale the data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        data_training_array = scaler.fit_transform(data_training)

        # load my my model
        model = load_model('model.h5')

        # make the prediction - testing part
        past_60_days = data_training.tail(60)
        final_df = past_60_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        # create test list
        x_test = []
        y_test = []
        # selec the range for test data
        for i in range(60, input_data.shape[0]):
            x_test.append(input_data[i - 60: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # make predictin
        y_predicted = model.predict(x_test)

        scaler = scaler.scale_

        # scale the data
        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # fancy plot
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='Test Price', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=list(range(len(y_predicted))), y=y_predicted.reshape(-1), mode='lines', name='Predicted Price', line=dict(color='red')))

        fig2.update_layout(title='Test Price vs Predicted Price',
                        xaxis_title='Time',
                        yaxis_title='Price',
                        xaxis_rangeslider_visible=True)

        st.plotly_chart(fig2)

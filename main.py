import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from logger.log import log_debug
import ta


def calculate_SMA(data, window=10):
    return data.rolling(window).mean()


def calculate_RSI(data, window=14):
    return ta.momentum.RSIIndicator(data['Close'], window).rsi()


def predictPrice(ticker="GBPUSD=X"):
    data = yf.download(ticker, period="1y", interval="1d")

    # Compute the target variable
    data = data.copy()
    data['Target'] = data['Close'].shift(-1)

    # Drop the last row as it will not have a target
    data = data[:-1].copy()

    # Get the new last row
    last_row = data.iloc[-1]

    # Print the date of the new last row
    log_debug(f"last date: {last_row.name}")

    # Create features based on the previous 3 candles
    for i in range(1, 4):
        data[f'Prev_Close_{i}'] = data['Close'].shift(i)
        data[f'Prev_Open_{i}'] = data['Open'].shift(i)
        data[f'Prev_High_{i}'] = data['High'].shift(i)
        data[f'Prev_Low_{i}'] = data['Low'].shift(i)
        data[f'Prev_Volume_{i}'] = data['Volume'].shift(i)

    # Add SMA and RSI features
    data['SMA'] = calculate_SMA(data['Close'])
    data['RSI'] = calculate_RSI(data)

    # Drop missing values
    data = data.dropna()

    # Split Data into Train and Test Sets
    X = data[['Prev_Close_1', 'Prev_Close_2', 'Prev_Close_3',
              'Prev_Open_1', 'Prev_Open_2', 'Prev_Open_3',
              'Prev_High_1', 'Prev_High_2', 'Prev_High_3',
              'Prev_Low_1', 'Prev_Low_2', 'Prev_Low_3',
              'Prev_Volume_1', 'Prev_Volume_2', 'Prev_Volume_3',
              'SMA', 'RSI']].values
    y = data['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Model
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train, y_train)

    # Predict the Close Price of the Next Candlestick
    next_candle_features = data[['Prev_Close_1', 'Prev_Close_2', 'Prev_Close_3',
                                'Prev_Open_1', 'Prev_Open_2', 'Prev_Open_3',
                                 'Prev_High_1', 'Prev_High_2', 'Prev_High_3',
                                 'Prev_Low_1', 'Prev_Low_2', 'Prev_Low_3',
                                 'Prev_Volume_1', 'Prev_Volume_2', 'Prev_Volume_3',
                                 'SMA', 'RSI']].iloc[-1:].values
    next_candle_prediction = regr.predict(next_candle_features)

    # Evaluate the model
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    log_debug(f"{ticker} ==> Mean Squared Error on the Test Set: {mse}")

    last_close_price = data['Prev_Close_1'].iloc[-1]
    predicted_close_price = next_candle_prediction[0]
    percentage_change = ((predicted_close_price - last_close_price) / last_close_price) * 100

    # show accuracy
    # Calculate the actual direction of price change
    actual_direction = np.sign(y_test[1:] - y_test[:-1])
    # Calculate the predicted direction of price change
    predicted_direction = np.sign(y_pred[1:] - y_pred[:-1])
    # Calculate the number of times the predicted direction matches the actual direction
    correct_directions = np.sum(predicted_direction == actual_direction)
    # Calculate accuracy
    accuracy = correct_directions / len(actual_direction)
    log_debug(f"Accuracy : {accuracy * 100:.2f}%")

    if predicted_close_price > last_close_price:
        log_debug(
            f"{ticker} ==> Predicted to be up. Price: {predicted_close_price}({percentage_change:.2f}%).")
    else:
        log_debug(
            f"{ticker} ==> Predicted to be down. Price: {predicted_close_price} ({percentage_change:.2f}%).")


def main():
    tickers = ["GBPUSD=X", "JPY=X", "NZDUSD=X", 'EURUSD=X']
    for ticker in tickers:
        predictPrice(ticker)


if __name__ == "__main__":
    main()

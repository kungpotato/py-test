import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from logger.log import log_debug, log_info, log_warning, log_error
import ta


def calculate_SMA(data, window=10):
    return data.rolling(window).mean()


def calculate_RSI(data, window=14):
    return ta.momentum.RSIIndicator(data['Close'], window).rsi()


def split_data(data, features, test_size=0.2):
    X = data[features].values
    y = data['Target'].values
    return train_test_split(X, y, test_size=test_size, random_state=42)


def train_model(X_train, y_train):
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train, y_train)
    return regr


def predict_next_candle(regr, features, last_data_point):
    next_candle_features = last_data_point[features].iloc[-1:].values
    return regr.predict(next_candle_features)


def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    actual_direction = np.sign(y_test[1:] - y_test[:-1])
    predicted_direction = np.sign(y_pred[1:] - y_pred[:-1])
    correct_directions = np.sum(predicted_direction == actual_direction)
    accuracy = correct_directions / len(actual_direction)
    return r2, accuracy


def log_prediction(last_close_price, predicted_close_price, percentage_change):
    if predicted_close_price > last_close_price:
        log_info(f"Predicted to be up. Price: {predicted_close_price}({percentage_change:.2f}%).")
    else:
        log_error(
            f"Predicted to be down. Price: {predicted_close_price} ({percentage_change:.2f}%).")


def calculate_consecutive_errors(y_test, y_pred):
    consecutive_errors = 0
    max_consecutive_errors = 0
    for i in range(1, len(y_test)):
        actual_direction = y_test[i] - y_test[i-1]
        predicted_direction = y_pred[i] - y_pred[i-1]
        if np.sign(actual_direction) != np.sign(predicted_direction):
            consecutive_errors += 1
        else:
            if consecutive_errors > 1:
                max_consecutive_errors += 1
            consecutive_errors = 0
    if consecutive_errors > 1:
        max_consecutive_errors += 1
    return max_consecutive_errors


def predictPrice(ticker):
    data = yf.download(ticker, period="5y", interval="1d")

    # Compute the target variable
    data = data.copy()
    data['Target'] = data['Close'].shift(-1)

    # Drop the last row as it will not have a target
    data = data[:-1].copy()

    # Get the new last row
    last_row = data.iloc[-1]

    log_warning(f'================================== {ticker} ==================================')

    # Print the date of the new last row
    log_debug(f"last date: {last_row.name}")

    # Create features based on the previous 3 candles
    for i in range(1, 11):
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

    # Prepare features and labels
    features = ['Prev_Close_1', 'Prev_Close_2', 'Prev_Close_3', 'Prev_Close_4', 'Prev_Close_5', 'Prev_Close_6', 'Prev_Close_7', 'Prev_Close_8', 'Prev_Close_9', 'Prev_Close_10',
                'Prev_Open_1', 'Prev_Open_2', 'Prev_Open_3', 'Prev_Open_4', 'Prev_Open_5', 'Prev_Open_6', 'Prev_Open_7', 'Prev_Open_8', 'Prev_Open_9', 'Prev_Open_10',
                'Prev_High_1', 'Prev_High_2', 'Prev_High_3', 'Prev_High_4', 'Prev_High_5', 'Prev_High_6', 'Prev_High_7', 'Prev_High_8', 'Prev_High_9', 'Prev_High_10',
                'Prev_Low_1', 'Prev_Low_2', 'Prev_Low_3', 'Prev_Low_4', 'Prev_Low_5', 'Prev_Low_6', 'Prev_Low_7', 'Prev_Low_8', 'Prev_Low_9', 'Prev_Low_10',
                'Prev_Volume_1', 'Prev_Volume_2', 'Prev_Volume_3', 'Prev_Volume_4', 'Prev_Volume_5', 'Prev_Volume_6', 'Prev_Volume_7', 'Prev_Volume_8', 'Prev_Volume_9', 'Prev_Volume_10',
                'SMA', 'RSI']

    X_train, X_test, y_train, y_test = split_data(data, features)
    regr = train_model(X_train, y_train)
    y_pred = regr.predict(X_test)
    next_candle_prediction = predict_next_candle(regr, features, data)
    r2, accuracy = evaluate_model(y_test, y_pred)
    last_close_price = data['Prev_Close_1'].iloc[-1]
    predicted_close_price = next_candle_prediction[0]
    percentage_change = ((predicted_close_price - last_close_price) / last_close_price) * 100
    log_prediction(last_close_price, predicted_close_price, percentage_change)
    max_consecutive_errors = calculate_consecutive_errors(y_test, regr.predict(X_test))

    log_debug(f'R-squared: {r2}')
    log_debug(f"Accuracy : {accuracy * 100:.2f}%")
    log_debug(f"Consecutive error more than once: {max_consecutive_errors}")


def main():
    tickers = ["GBPUSD=X", 'GBPJPY=X', "JPY=X", 'EURUSD=X',
               "NZDUSD=X", 'AUDUSD=X', 'GC=F']
    for ticker in tickers:
        predictPrice(ticker)


if __name__ == "__main__":
    main()

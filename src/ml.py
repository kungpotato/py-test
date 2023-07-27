import yfinance as yf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

# Download historical data
ticker = "AOT.BK"
data = yf.download(ticker,interval='1d',period='max')

# Calculate 20-day and 60-day EMA
data['20_EMA'] = data['Close'].ewm(span=12).mean()
data['60_EMA'] = data['Close'].ewm(span=24).mean()

# Calculate RSI
rsi_indicator = RSIIndicator(data['Close'])
data['RSI'] = rsi_indicator.rsi()

# Shift the data to predict 3 months into the future (approx. 63 trading days)
data['target_price'] = data['Close'].shift(-63)

# Drop the last 63 rows where we don't know the future close price
data = data.dropna()

# Split into train and test set
X = data[['20_EMA', '60_EMA','RSI', 'Volume']]
y = data['target_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVR()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Print the predicted price for the last date in the test set
print(f"Predicted price for {X_test.index[-1]}: ${predictions[-1]}")

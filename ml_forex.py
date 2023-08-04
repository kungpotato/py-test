import yfinance as yf
import pandas as pd

# List of tickers
tickers = ["GBPUSD=X", "JPY=X", "NZDUSD=X", 'EURUSD=X', 'GBPJPY=X', 'AUDUSD=X']

# Empty DataFrame to store the closing prices
closing_prices = pd.DataFrame()

# Download data for each ticker and store the closing prices
for ticker in tickers:
    data = yf.download(ticker, period="1y", interval="1d")
    closing_prices[ticker] = data['Close']

# Compute the correlation matrix
correlation_matrix = closing_prices.corr()

# Unstack the matrix and sort by correlation
correlations = correlation_matrix.unstack().sort_values(ascending=False)

# Remove self-correlations (correlation of a ticker with itself is always 1)
correlations = correlations[correlations != 1]

# Remove duplicate pairs
correlations = correlations.iloc[::2]

# Print pairs of tickers from most to least correlated
print("Pairs of tickers from most to least correlated:")
for pair, correlation in correlations.items():
    print(f"{pair}: {correlation}")

import yfinance as yf
ticker = "GBPUSD=X"

data = yf.download(ticker, period="1y", interval="1d")

last_row = data.iloc[-1]

# Print the date and close price
print("Date:", last_row.name)
print("Close price:", last_row['Close'])

# Remove the last row
data = data.iloc[:-1]

# Get the new last row
new_last_row = data.iloc[-1]

# Print the date of the new last row
print("New last date:", new_last_row.name)

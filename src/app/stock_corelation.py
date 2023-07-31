import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the tickers of the stocks you want to analyze
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Download historical stock price data
data = yf.download(tickers, start="2022-01-01", end="2023-07-27")

# Use only Close prices for correlation
data = data['Close']

# Calculate the correlation matrix
corr_matrix = data.corr()

# Unstack the correlation matrix into a Series
corr_pairs = corr_matrix.unstack()

# Sort the correlation pairs by correlation strength in descending order
sorted_pairs = corr_pairs.sort_values(key=abs, ascending=False)

# Filter out the pairs that have a correlation of 1 (i.e., a stock with itself)
non_identical_pairs = sorted_pairs[sorted_pairs != 1]

# Print the pairs of stocks that are most strongly correlated
print("Most strongly correlated pairs:")
print(non_identical_pairs.head(10))

# Sort the correlation pairs by correlation strength in ascending order (weaker correlations first)
sorted_pairs_asc = corr_pairs.sort_values(key=abs, ascending=True)

# Filter out the pairs that have a correlation of 1 (i.e., a stock with itself)
non_identical_pairs_asc = sorted_pairs_asc[sorted_pairs_asc != 1]

# Print the pairs of stocks that are least correlated
print("\nLeast correlated pairs:")
print(non_identical_pairs_asc.head(10))

# Plot the correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

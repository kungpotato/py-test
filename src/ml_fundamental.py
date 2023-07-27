import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# # Random seed for reproducibility
# np.random.seed(0)

# # Number of data points
# n = 100

# # Create a dictionary with the columns as keys and random data as values
# data = {
#     'EPS': np.random.normal(5, 1, n),
#     'PE_Ratio': np.random.normal(20, 5, n),
#     'PB_Ratio': np.random.normal(1.5, 0.5, n),
#     'PS_Ratio': np.random.normal(2.5, 0.5, n),
#     'Dividend_Yield': np.random.normal(2, 0.5, n),
#     'Market_Cap': np.random.normal(50000000000, 10000000000, n),
#     'Debt_Equity_Ratio': np.random.normal(0.5, 0.1, n),
#     'ROE': np.random.normal(20, 5, n),
#     'Current_Ratio': np.random.normal(1.2, 0.2, n),
#     'Operating_Margin': np.random.normal(25, 5, n),
#     'Stock_Price': np.random.normal(100, 20, n)
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Save the DataFrame to a csv file
# df.to_csv('stock_data.csv', index=False)

# Load the dataset
df = pd.read_csv('stock_data.csv')

# Features
X = df[['EPS', 'PE_Ratio', 'PB_Ratio', 'PS_Ratio', 'Dividend_Yield', 'Market_Cap', 'Debt_Equity_Ratio', 'ROE', 'Current_Ratio', 'Operating_Margin']]
# Target variable
y = df['Stock_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict future stock price
future_stock_data = {'EPS': [5.20], 'PE_Ratio': [15.7], 'PB_Ratio': [1.6], 'PS_Ratio': [2.8], 'Dividend_Yield': [1.5], 'Market_Cap': [50000000000], 'Debt_Equity_Ratio': [0.5], 'ROE': [20], 'Current_Ratio': [1.2], 'Operating_Margin': [25]}
future_stock_df = pd.DataFrame(future_stock_data)
future_stock_price = model.predict(future_stock_df)
print(f'Predicted future stock price: {future_stock_price[0]}')
import yfinance as yf
import numpy as np

# Fetch data
ticker_symbol = 'AAPL'
ticker_data = yf.Ticker(ticker_symbol)
ticker_df = ticker_data.history(period='1y', interval='1d')

# Calculate whether the price went up or down
ticker_df['Direction'] = (ticker_df['Close'] > ticker_df['Open']).astype(int)

# Define the states, actions, and Q-table
states = len(ticker_df)
actions = 2  # up or down
Q_table = np.zeros((states, actions))

# Hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate

# Training loop
for i in range(states - 1):
    state = i
    next_state = i + 1

    # Choose action based on epsilon-greedy policy
    if np.random.rand() < epsilon:
        action = np.random.choice(actions)
    else:
        action = np.argmax(Q_table[state])

    # Get reward based on correctness of prediction
    correct_prediction = (action == ticker_df['Direction'].iloc[next_state])
    reward = 1 if correct_prediction else 0

    # Q-learning update
    best_next_action = np.argmax(Q_table[next_state])
    Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * \
        (reward + gamma * Q_table[next_state, best_next_action])

# Making prediction for the last state
state = states - 1
action = np.argmax(Q_table[state])
prediction = "Up" if action == 1 else "Down"

print(f"The prediction for the next candlestick is: {prediction}")

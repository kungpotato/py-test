import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator

stock = 'PTT.BK'
# stock = "BTC-USD"

# Download intraday data
data = yf.download(tickers=stock, period='2y', interval='1d')

# Calculate moving average and RSI
data['SMA1'] = SMAIndicator(data['Close'], window=40).sma_indicator()
data['SMA2'] = SMAIndicator(data['Close'], window=120).sma_indicator()
data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()

# Fundamental
company_info = yf.Ticker(stock).info
pe_ratio = company_info['pegRatio']
data['peg'] = pe_ratio

# Define a signal (buy=1 , sell=-1, do nothing=0)
data['long_entry'] = (data['SMA1'] > data['SMA2']) & (data['peg']<1)
data['long_exit'] = (data['SMA1'] < data['SMA2'])
data.loc[data['long_entry'], 'signal'] = 1
data.loc[data['long_exit'], 'signal'] = -1

# Forward fill signals
data['signal'].ffill(inplace=True)

# Calculate returns
data['return'] = data['Close'].pct_change() * data['signal'].shift()

# Calculate cumulative returns
data['cumulative_return'] = (data['return'] + 1).cumprod()

# Calculate drawdown
data['cummax'] = data['cumulative_return'].cummax()
data['drawdown'] = data['cummax'] - data['cumulative_return']

# Calculate risk-reward (assuming risk-free rate is 0)
data['risk_reward'] = data['cumulative_return'] / data['drawdown'].where(data['drawdown'] != 0)

# Show entries and exits
data['entry'] = data['signal'] > 0
data['exit'] = data['signal'] < 0

# Plotting candlestick chart with entry and exit points
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'], high=data['High'],
                                     low=data['Low'], close=data['Close'])])

fig.add_trace(go.Scatter(x=data[data['entry']].index, y=data[data['entry']]['Close'],
                         mode='markers', marker=dict(color='green', size=10), name='Entry'))

fig.add_trace(go.Scatter(x=data[data['exit']].index, y=data[data['exit']]['Close'],
                         mode='markers', marker=dict(color='red', size=10), name='Exit'))

fig.update_layout(title=f'{stock} Trading with SMA',
                  xaxis_title='Date', yaxis_title='Price', template='plotly_dark')

fig.show()

# summary statistics
total_return = data['cumulative_return'].iloc[-1] - 1
positive_return = data[data['return'] > 0]['return'].sum()
negative_return = data[data['return'] < 0]['return'].sum()
win_rate = len(data[data['return'] > 0]) / len(data[data['return'].notnull()])
max_drawdown = data['drawdown'].max()
average_risk_reward = data['risk_reward'].mean()

# print(f'Total return: {total_return}')
# print(f'Positive return: {positive_return}')
# print(f'Negative return: {negative_return}')
# print(f'Win rate: {win_rate}')
# print(f'Max Drawdown: {max_drawdown}')
# print(f'Average Risk/Reward: {average_risk_reward}')

# Create a dictionary with stats
stats = {'Total return': total_return * 100,
         'Positive return': positive_return * 100,
         'Negative return': negative_return * 100,
         'Win rate': win_rate * 100,
         'Max Drawdown': max_drawdown,
         'Average Risk/Reward': average_risk_reward}

# Convert the dictionary to a pandas DataFrame
stats_df = pd.DataFrame(list(stats.items()), columns=['Stat', 'Value'])

# Plot stats
plt.figure(figsize=(10, 6))
bars = plt.barh(stats_df['Stat'], stats_df['Value'], color='skyblue')
# Add value annotations to each bar
for bar in bars:
    width = bar.get_width() # Get bar width
    label_y_pos = bar.get_y() + bar.get_height() / 2
    plt.text(width, label_y_pos, f'{width:.2f}%', va='center')

plt.title('Trading Statistics')
plt.xlabel('Value (%)')
plt.ylabel('Statistic')
plt.grid(axis='x')


# Calculate number of winning and losing trades
n_wins = len(data[data['return'] > 0])
n_losses = len(data[data['return'] < 0])

# Create a pie chart
labels = ['Wins', 'Losses']
sizes = [n_wins, n_losses]
colors = ['green', 'red']
explode = (0.1, 0)  # explode 1st slice (i.e., 'Wins')

# Plot
plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Winning vs Losing Trades')

plt.show()


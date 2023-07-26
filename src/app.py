import yfinance as yf
import ta
import pandas as pd
import plotly.graph_objs as go

# Download historical data as dataframe
ticker = "AAPL"
df = yf.download(ticker, interval='1d', period='1y')

# Calculate the short-window simple moving average
short_sma = 50
df['SMA1'] = ta.trend.sma_indicator(df['Close'], window=short_sma)

# Calculate the long-window simple moving average
long_sma = 200
df['SMA2'] = ta.trend.sma_indicator(df['Close'], window=long_sma)

# Create a trace for the short-window simple moving average
trace_sma1 = go.Scatter(
    x=df.index,
    y=df['SMA1'],
    mode='lines',
    name=f'SMA{short_sma}'
)

# Create a trace for the long-window simple moving average
trace_sma2 = go.Scatter(
    x=df.index,
    y=df['SMA2'],
    mode='lines',
    name=f'SMA{long_sma}'
)

# Create a Candlestick trace
trace_candle = go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Candlestick'
)

# Define the entry points where short moving average crosses above long moving average
df['Buy_Signal'] = (df['SMA1'] > df['SMA2']) & (
    df['SMA1'].shift(1) < df['SMA2'].shift(1))

# Define the exit points where short moving average crosses below long moving average
df['Sell_Signal'] = (df['SMA1'] < df['SMA2']) & (
    df['SMA1'].shift(1) > df['SMA2'].shift(1))

# Create traces for the entry and exit points
trace_buy = go.Scatter(
    x=df[df['Buy_Signal']].index,
    y=df[df['Buy_Signal']]['Close'],
    mode='markers',
    name='Buy',
    marker=dict(
        symbol='triangle-up',
        color='green',
        size=10
    )
)

trace_sell = go.Scatter(
    x=df[df['Sell_Signal']].index,
    y=df[df['Sell_Signal']]['Close'],
    mode='markers',
    name='Sell',
    marker=dict(
        symbol='triangle-down',
        color='red',
        size=10
    )
)


data = [trace_candle, trace_sma1, trace_sma2, trace_buy, trace_sell]

layout = go.Layout(
    title=f'{ticker} Candlestick with SMA',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price'),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.0
    ),
    template='plotly_dark'
)

fig = go.Figure(data=data, layout=layout)

fig.show()

import numpy as np
import yfinance as yf
import ta
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import linregress
import matplotlib.pyplot as plt

class StockAnalysis:
    def __init__(self, tricker, period='max', interval='1d', indicator='sma',risk_free_rate=0.015):
        self.__tricker = tricker
        self.__period = period
        self.__interval = interval
        self.indicator = indicator
        self.__risk_free_rate = risk_free_rate
        self.__df = self.__get_data()

    def __get_data(self):
        df = yf.download(
            self.__tricker, interval=self.__interval, period=self.__period)
        return df

    def sma_plot(self, short_sma=20, long_sma=60):
        df = self.__df
        df['SMA1'] = ta.trend.sma_indicator(df['Close'], window=short_sma)
        df['SMA2'] = ta.trend.sma_indicator(df['Close'], window=long_sma)
        df['Buy_Signal'] = (df['SMA1'] > df['SMA2']) & (
            df['SMA1'].shift(1) < df['SMA2'].shift(1))
        df['Sell_Signal'] = (df['SMA1'] < df['SMA2']) & (
            df['SMA1'].shift(1) > df['SMA2'].shift(1))

        traces = [
            go.Scatter(x=df.index, y=df['SMA1'], mode='lines', name='SMA40'),
            go.Scatter(x=df.index, y=df['SMA2'], mode='lines', name='SMA120'),
            go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                           low=df['Low'], close=df['Close'], name='Candlestick'),
            go.Scatter(x=df[df['Buy_Signal']].index, y=df[df['Buy_Signal']]['Close'], mode='markers',
                       name='Buy', marker=dict(symbol='triangle-up', color='green', size=20)),
            go.Scatter(x=df[df['Sell_Signal']].index, y=df[df['Sell_Signal']]['Close'],
                       mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=20))
        ]

        layout = go.Layout(title=f'{self.__tricker} Candlestick with SMA', xaxis=dict(title='Date'), yaxis=dict(
            title='Price'), showlegend=True, legend=dict(x=0, y=1.0), template='plotly_dark')
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def calculate_returns(self):
        self.__df['Returns'] = self.__df['Close'].pct_change()

    def calculate_sharpe_ratio(self):
        self.calculate_returns()
        excess_returns = self.__df['Returns'] - self.__risk_free_rate/252  # Assuming 252 trading days in a year
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self):
        self.calculate_returns()
        rolling_max = self.__df['Returns'].cummax()
        drawdown = self.__df['Returns'] - rolling_max
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_win_rate(self):
        self.__df['Win'] = np.where(self.__df['Returns'] > 0, 1, 0)
        win_rate = self.__df['Win'].sum() / self.__df.shape[0]
        return win_rate

    def calculate_profit_factor(self):
        gross_profit = self.__df[self.__df['Returns'] > 0]['Returns'].sum()
        gross_loss = self.__df[self.__df['Returns'] < 0]['Returns'].sum()
        if gross_loss == 0: return np.inf
        return gross_profit / gross_loss

    def calculate_avg_profit_per_trade(self):
        avg_profit_per_trade = self.__df['Returns'].mean()
        return avg_profit_per_trade

    def calculate_beta(self, benchmark_returns):
        # Calculate beta compared to a benchmark's returns
        self.calculate_returns()
        # Remove NaN values
        self.__df.dropna(subset=['Returns'], inplace=True)
        benchmark_returns.dropna(inplace=True)
        # Perform linear regression
        beta, alpha, r_value, p_value, std_err = linregress(
            benchmark_returns.values,
            self.__df['Returns'].values
        )
        return beta


def main():
    my_stock = StockAnalysis(tricker='sc.bk',period='max',interval='1wk')
    my_stock.sma_plot()
    sharpe_ratio = my_stock.calculate_sharpe_ratio()
    max_drawdown = my_stock.calculate_max_drawdown()
    win_rate = my_stock.calculate_win_rate()
    profit_factor = my_stock.calculate_profit_factor()
    avg_profit_per_trade = my_stock.calculate_avg_profit_per_trade()

    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Maximum Drawdown: {max_drawdown}")
    print(f"Win Rate: {win_rate}")
    print(f"Profit Factor: {profit_factor}")
    print(f"Average Profit per Trade: {avg_profit_per_trade}")

    # Plotting
    metrics = ['Sharpe Ratio', 'Maximum Drawdown', 'Win Rate', 'Profit Factor', 'Average Profit per Trade']
    values = [sharpe_ratio, max_drawdown, win_rate, profit_factor, avg_profit_per_trade]

    fig, ax = plt.subplots()
    ax.bar(metrics, values, color='blue')
    plt.xticks(rotation=45)
    ax.set_title('Stock Analysis Metrics')
    plt.show()



if __name__ == "__main__":
    main()

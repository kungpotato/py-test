import numpy as np
import yfinance as yf
import ta
import pandas as pd
import plotly.graph_objs as go


class StockAnalysis:
    def __init__(self, tricker, period='1y', interval='1d', indicator='sma'):
        self.__tricker = tricker
        self.__period = period
        self.__interval = interval
        self.indicator = indicator
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

    def calc_win_rate_and_RRR(self):
        df = self.__df
        buy_prices = df[df['Buy_Signal']]['Close']
        sell_prices = df[df['Sell_Signal']]['Close']

        if len(buy_prices) > len(sell_prices):
            buy_prices = buy_prices.iloc[:len(sell_prices)]
        elif len(sell_prices) > len(buy_prices):
            sell_prices = sell_prices.iloc[:len(buy_prices)]

        gains = sell_prices.values - buy_prices.values

        profitable_trades = gains[gains > 0]
        win_rate = len(profitable_trades) / len(gains)
        print(f'Win Rate: {win_rate*100:.2f}%')

        losses = -gains[gains < 0]
        wins = gains[gains > 0]

        average_loss = losses.mean() if len(losses) > 0 else 0
        average_gain = wins.mean() if len(wins) > 0 else 0

        RRR = average_gain / average_loss if average_loss > 0 else float('inf')
        print(f'Risk-Reward Ratio: {RRR:.2f}')


def main():
    my_stock = StockAnalysis('aot.bk')
    my_stock.sma_plot()
    my_stock.calc_win_rate_and_RRR()


if __name__ == "__main__":
    main()

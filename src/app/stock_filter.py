import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_sp500_tickers():
    resp = requests.get('http://siamchart.com/stock/')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'tbl'})
    tickers = []
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if cols:
            first_col_data = cols[0].get_text()
            tickers.append(first_col_data)
    return tickers

def filter_stocks(tickers):
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            pe = stock.info['pegRatio']
            if pe < 1:
                data.append([ticker, pe])
        except:
            continue

    df = pd.DataFrame(data, columns=['Ticker', 'PEG Ratio'])
    return df

# Get the list of S&P 500 tickers
tickers = scrape_sp500_tickers()

# Filter the stocks with PE ratio
filtered_stocks = filter_stocks(tickers)
print(filtered_stocks)

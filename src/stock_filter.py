import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)
    return tickers

def filter_stocks(tickers, pe_min, pe_max):
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            pe = stock.info['trailingPE']
            if pe_min <= pe <= pe_max:
                data.append([ticker, pe])
        except:
            continue

    df = pd.DataFrame(data, columns=['Ticker', 'PE Ratio'])
    return df

# Get the list of S&P 500 tickers
tickers = scrape_sp500_tickers()

# Filter the stocks with PE ratio between 10 and 20
filtered_stocks = filter_stocks(tickers, 10, 20)
print(filtered_stocks)

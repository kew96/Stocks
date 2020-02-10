import requests
import pandas
from alpha_vantage.timeseries import TimeSeries

alpha_vantage_KEY = '7ZDI2M6PEWCEOSFC'
av_ts = TimeSeries(alpha_vantage_KEY)


def ticker_search(name):
    return av_ts.get_symbol_search(name)


class Stock:

    def __init__(self, ticker=None, name=None):
        self.ticker = ticker
        self.name = self.get_name() if name is None else name
        # TODO: self.name
        self.data, self.meta = av_ts.get_daily_adjusted(symbol=self.ticker, outputsize='full')
        # av_ts.

    def get_name(self):
        if self.ticker is None:
            pass

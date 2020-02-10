from alpha_vantage.timeseries import TimeSeries
from .errors import *

alpha_vantage_KEY = '7ZDI2M6PEWCEOSFC'
av_ts = TimeSeries(alpha_vantage_KEY)


def ticker_search(name):
    return av_ts.get_symbol_search(name)


class Stock:

    def __init__(self, ticker=None, name=None):
        self.ticker = ticker
        self.name = name
        self.__set_name()
        # TODO: self.name
        self.data, self.meta = av_ts.get_daily_adjusted(symbol=self.ticker, outputsize='full')
        # av_ts.

    def __set_name(self):
        if self.ticker is None and self.name is None:
            raise NoTickerError()
        elif self.ticker is not None:
            vals = ticker_search(self.ticker)
        else:
            vals = ticker_search(self.name)
        if vals[0]['9. matchScore'] > 0.7:
            self.ticker = vals[0]['1. symbol']
            self.name = vals[0]['2. name']

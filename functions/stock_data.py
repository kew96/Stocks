from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from errors import *
from alias import *

alpha_vantage_KEY = '7ZDI2M6PEWCEOSFC'
av_ts = TimeSeries(alpha_vantage_KEY)
av_ti = TechIndicators(alpha_vantage_KEY, output_format='pandas')


def ticker_search(name):
    return av_ts.get_symbol_search(name)


@aliased
class Stock:

    def __init__(self, ticker=None, name=None, size='compact'):
        self.ticker = ticker
        self.name = name
        self.__set_name_ticker()
        self.size = size
        self.__check_size()
        self.now_data, self.now_meta = av_ts.get_intraday(symbol=self.ticker, outputsize=size, interval='1min')
        self.daily_data, self.daily_meta = av_ts.get_daily_adjusted(symbol=self.ticker, outputsize=size)

    def __set_name_ticker(self):
        if self.ticker is None and self.name is None:
            raise NoTickerError()
        elif self.ticker is not None:
            vals = ticker_search(self.ticker)[0]
        else:
            vals = ticker_search(self.name)[0]
        if float(vals[0]['9. matchScore']) > 0.7:
            self.ticker = vals[0]['1. symbol']
            self.name = vals[0]['2. name']
        elif float(vals[0]['9. matchScore'] > 0.3):
            choices = []
            for c in vals:
                if float(c['9. matchScore']) > 0.3:
                    choices.append(c)
            string = 'Please select one of the following choices by number:'
            for i, c in enumerate(choices):
                substring = ''
                for key, val in c.items():
                    if key[0] in ['1', '2', '3', '4', '8', '9']:
                        substring = substring + '\t\t' + key[3:] + ': ' + val + ', '
                string = string + '\n' + str(i + 1) + '. ' + substring[:-2]
            pick = input(string + '\n' + 'Input: ')
            pick = choices[int(pick) - 1]
            self.ticker = pick['1. symbol']
            self.name = pick['2. name']
        else:
            raise NoTickerError()

    def __check_size(self):

    # TODO: check that size is either 'compact' or 'full'

    @Alias('sma', 'SMA')
    def simple_moving_average(self, period='daily', data_type='close'):
        period_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        type_options = ['close', 'open', 'high', 'low']
        if period not in period_options:
            print('"' + period + '" is not a valid period.')
            output = 'Please select one of the following period options: '
            for o in period_options:
                output = output + o + ' '
            period = input(output)
        if data_type not in type_options:
            print('"' + data_type + '" is not a valid data point type.')
            output = 'Please select one of the following data point type options: '
            for o in type_options:
                output = output + o + ' '
            data_type = input(output)
        return av_ti.get_sma(symbol=self.ticker, interval=period, time_period=200, series_type=data_type)


if __name__ == '__main__':
    s = Stock('AAPL')
    print(s.sma('1min'))

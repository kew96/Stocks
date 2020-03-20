from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from errors import *
from alias import *
import numpy as np

alpha_vantage_KEY = '7ZDI2M6PEWCEOSFC'
av_ts = TimeSeries(alpha_vantage_KEY)
av_ti = TechIndicators(alpha_vantage_KEY, output_format='pandas')


def ticker_search(name):
    return av_ts.get_symbol_search(name)


@aliased
class Stock:

    def __init__(self, ticker=None, name=None, start=None, end=None):
        self.ticker = ticker
        self.name = name
        self.start = start
        self.end = end
        self.historical = False if start is None and end is None else True
        self.__set_name_ticker()
        # self.__check_size()
        # self.intra_data, self.intra_meta = av_ts.get_intraday(symbol=self.ticker, outputsize=size, interval='1min')
        # self.daily_data, self.daily_meta = av_ts.get_daily_adjusted(symbol=self.ticker, outputsize=size)
        # self.current, _ = av_ts.get_quote_endpoint(symbol=self.ticker)

    def __str__(self):
        if self.historical:
            return self.ticker + f' Historical:{self.historical} ({self.start}-{self.end}'
        return self.ticker

    def __set_name_ticker(self):
        if self.ticker is None and self.name is None:
            raise NoTickerError
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
            raise NoTickerError

    @Alias('sma', 'SMA')
    def simple_moving_average(self):  # TODO: implement
        pass

    @Alias('ema', 'EMA')
    def exponential_moving_average(self):  # TODO: implement
        pass

    # TODO: WMA

    # TODO: DEMA

    # TODO: TEMA

    # TODO: TRIMA

    # TODO: KAMA

    # TODO: MAMA

    @Alias('vwap', 'VWAP')
    def volume_weighted_average_price(self):  # TODO: implement
        pass

    # TODO: T3

    @Alias('macd', 'MACD')
    def moving_average_convergence_divergence(self):  # TODO: implement
        pass

    # TODO: MACDEXT

    @Alias('stoch', 'STOCH', 'stoch_oscillator')
    def stochastic_oscillator(self):  # TODO: implement
        pass

    # TODO: STOCHF

    @Alias('rsi', 'RSI', 'relative_strength')
    def relative_strength_index(self):  # TODO: implement
        pass

    # TODO: STOCHRSI

    # TODO: WILLR

    @Alias('adx', 'ADX', 'average_directional_movement')
    def average_directional_movement_index(self):  # TODO: implement
        pass

    # TODO: ADXR

    # TODO: APO

    # TODO: PPO

    # TODO: MOM

    # TODO: BOP

    @Alias('cci', 'CCI', 'commodity_channel')
    def commodity_channel_index(self):  # TODO: implement
        pass

    # TODO: CMO

    # TODO: ROC

    # TODO: ROCR

    @Alias('Aroon', 'AROON')
    def aroon(self):  # TODO: implement
        pass

    # TODO: AROONOSC

    # TODO: MFI

    # TODO: TRIX

    # TODO: ULTOSC

    # TODO: DX

    # TODO: MINUS_DI

    # TODO: PLUS_DI

    # TODO: MINUS_DM

    # TODO: PLUS_DM

    @Alias('bbands', 'BBANDS', 'Bollinger_bands')
    def bollinger_bands(self):  # TODO: implement
        pass

    # TODO: MIDPOINT

    # TODO: MIDPRICE

    # TODO: SAR

    # TODO: TRANGE

    # TODO: ATR

    # TODO: NATR

    @Alias('ad', 'AD', 'Chaikin_AD_Line', 'Chaikin_AD_line', 'chaikin_ad_line')
    def chaikin_ad_line_values(self):  # TODO: implement
        pass

    # TODO: ADOSC

    @Alias('obv', 'OBV', 'balance_volume')
    def on_balance_volume(self):  # TODO: implement
        pass

    # TODO: HT_TRENDLINE

    # TODO: HT_SINE

    # TODO: HT_TRENDMODE

    # TODO: HT_DCPERIOD

    # TODO: HT_DCPHASE

    # TODO: HT_PHASOR


if __name__ == '__main__':
    s = Stock('AAPL')
    print(s.historical)

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
        self.size = self.size.lower()
        if self.size not in ['compact', 'full']:
            raise WrongFormatError(self.size)

    @staticmethod
    def __create_output_question(choice, options):
        print('"' + choice + '" is not a valid option.')
        output = 'Please choose one of the following options by number: \n'
        if type(options) == list:
            print('"' + choice + '" is not a valid option.')
            output = 'Please choose one of the following options by number: \n'
            for i, o in enumerate(options):
                output = output + str(i + 1) + '. ' + o + '\n'
            index = int(input(output)) - 1
            return options[index]
        elif type(options) == dict:
            pass

    @Alias('sma', 'SMA')
    def simple_moving_average(self, interval='daily', data_type='close'):
        """
        :param interval: a string defining the period of each average the options are:
                        ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        :param data_type: a string defining which values (for 'daily', 'weekly', 'monthly') to use.
                    The options are the following: ['close', 'open', 'high', 'low']
        :return: returns a dataframe with a datetime index and floats for values
        """
        period_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        type_options = ['close', 'open', 'high', 'low']
        if interval not in period_options:
            interval = self.__create_output_question(interval, period_options)
        if data_type not in type_options:
            data_type = self.__create_output_question(data_type, type_options)
        return av_ti.get_sma(symbol=self.ticker, interval=interval, time_period=200, series_type=data_type)

    @Alias('ema', 'EMA')
    def exponential_moving_average(self, interval='daily', data_type='close'):
        """
        :param interval: a string defining the period of each average the options are:
                        ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        :param data_type: a string defining which values (for 'daily', 'weekly', 'monthly') to use.
                    The options are the following: ['close', 'open', 'high', 'low']
        :return: returns a dataframe with a datetime index and floats for values
        """
        period_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        type_options = ['close', 'open', 'high', 'low']
        if interval not in period_options:
            interval = self.__create_output_question(interval, period_options)
        if data_type not in type_options:
            data_type = self.__create_output_question(data_type, type_options)
        return av_ti.get_ema(symbol=self.ticker, interval=interval, time_period=200, series_type=data_type)

    # TODO: WMA

    # TODO: DEMA

    # TODO: TEMA

    # TODO: TRIMA

    # TODO: KAMA

    # TODO: MAMA

    @Alias('vwap', 'VWAP')
    def volume_weighted_average_price(self, interval='15min'):
        period_options = ['1min', '5min', '15min', '30min', '60min']
        if interval not in period_options:
            interval = self.__create_output_question(interval, period_options)
        return av_ti.get_vwap(symbol=self.ticker, interval=interval)

    # TODO: T3

    @Alias('macd', 'MACD')
    def moving_average_convergence_divergence(self, interval='daily', data_type='close', fastperiod=12,
                                              slowperiod=26, signalperiod=9):
        pass
        # TODO: MACD*

    # TODO: MACDEXT

    # TODO: STOCH*
    @Alias('stoch', 'STOCH', 'stoch_oscillator')
    def stochastic_oscillator(self, interval='daily', fast_k_period=5, slow_k_period=3,
                              slow_d_period=3, slow_k_ma_type=0, slow_d_ma_type=0):
        type_options = {}
        if fast_k_period <= 0 or type(fast_k_period) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(fast_k_period))
            fast_k_period = int(choice)
        if slow_k_period <= 0 or type(slow_k_period) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(slow_k_period))
            slow_k_period = int(choice)
        if slow_d_period <= 0 or type(slow_d_period) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(slow_d_period))
            slow_d_period = int(choice)


# TODO: STOCHF

# TODO: RSI*

# TODO: STOCHRSI

# TODO: WILLR

# TODO: ADX*

# TODO: ADXR

# TODO: APO

# TODO: PPO

# TODO: MOM

# TODO: BOP

# TODO: CCI*

# TODO: CMO

# TODO: ROC

# TODO: ROCR

# TODO: AROON*

# TODO: AROONOSC

# TODO: MFI

# TODO: TRIX

# TODO: ULTOSC

# TODO: DX

# TODO: MINUS_DI

# TODO: PLUS_DI

# TODO: MINUS_DM

# TODO: PLUS_DM

# TODO: BBANDS*

# TODO: MIDPOINT

# TODO: MIDPRICE

# TODO: SAR

# TODO: TRANGE

# TODO: ATR

# TODO: NATR

# TODO: AD*

# TODO: ADOSC

# TODO: OBV*

# TODO: HT_TRENDLINE

# TODO: HT_SINE

# TODO: HT_TRENDMODE

# TODO: HT_DCPERIOD

# TODO: HT_DCPHASE

# TODO: HT_PHASOR


if __name__ == '__main__':
    s = Stock('AAPL')
    print(s.vwap('1'))

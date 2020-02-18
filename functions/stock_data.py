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
        if choice.lower() != 'help':
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
            for key, val in options.items():
                if str(val + 1) in output:
                    output = output[:-1] + ' (' + key + ')\n'
                else:
                    output = output + str(val + 1) + '. ' + key + '\n'
            index = int(input(output)) - 1
            return index

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
        period_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        type_options = ['close', 'open', 'high', 'low']
        if interval not in period_options:
            interval = self.__create_output_question(interval, period_options)
        if data_type not in type_options:
            data_type = self.__create_output_question(data_type, type_options)
        if fastperiod <= 0 or type(fastperiod) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(fastperiod))
            fastperiod = int(choice)
        if slowperiod <= 0 or type(slowperiod) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(slowperiod))
            slowperiod = int(choice)
        if signalperiod <= 0 or type(signalperiod) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(signalperiod))
            signalperiod = int(choice)
        return av_ti.get_macd(symbol=self.ticker, interval=interval, fastperiod=fastperiod,
                              slowperiod=slowperiod, signalperiod=signalperiod)
    # TODO: MACDEXT

    @staticmethod
    def __check_convert(entry, options):
        if type(entry) in (int, float):
            entry = int(entry)
        else:
            if ' ' in entry:
                if entry[:4].lower() == 'mesa':
                    entry = entry[:4].upper() + entry[4:].title()
                else:
                    entry = entry.title()
            else:
                entry = entry.upper()
            if entry in options.keys() or entry in options.values():
                return options[entry]
            else:
                return Stock.__create_output_question(entry, options)

    @Alias('stoch', 'STOCH', 'stoch_oscillator')
    def stochastic_oscillator(self, interval='daily', fast_k_period=5, slow_k_period=3,
                              slow_d_period=3, slow_k_ma_type='help', slow_d_ma_type='help'):
        type_options = {
            'Simple Moving Average': 0, 'SMA': 0, 'Exponential Moving Average': 1, 'EMA': 1,
            'Weighted Moving Average': 2, 'WMA': 2, 'Double Exponential Moving Average': 3, 'DEMA': 3,
            'Triple Exponential Moving Average': 4, 'TEMA': 4, 'Triangular Moving Average': 5, 'TRIMA': 5,
            'T3 Moving Average': 6, 'Kaufman Adaptive Moving Average': 7, 'KAMA': 7,
            'MESA Adaptive Moving Average': 8, 'MAMA': 8
        }
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        slow_k_ma_type = self.__check_convert(slow_k_ma_type, type_options)
        slow_d_ma_type = self.__check_convert(slow_d_ma_type, type_options)
        if fast_k_period <= 0 or type(fast_k_period) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(fast_k_period))
            fast_k_period = int(choice)
        if slow_k_period <= 0 or type(slow_k_period) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(slow_k_period))
            slow_k_period = int(choice)
        if slow_d_period <= 0 or type(slow_d_period) != int:
            choice = input('Please select a positive integer, {} is not a valid choice:\n'.format(slow_d_period))
            slow_d_period = int(choice)
        if interval not in interval_options:
            interval = self.__create_output_question(interval, interval_options)
        return av_ti.get_stoch(self.ticker, interval=interval, fastkperiod=fast_k_period, slowkperiod=slow_k_period,
                               slowdperiod=slow_d_period, slowkmatype=slow_k_ma_type, slowdmatype=slow_d_ma_type)

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
    print(s.stoch())

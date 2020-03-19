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

    def __init__(self, ticker=None, name=None, size='compact'):
        self.ticker = ticker
        self.name = name
        self.__set_name_ticker()
        self.size = size
        self.__check_size()
        self.intra_data, self.intra_meta = av_ts.get_intraday(symbol=self.ticker, outputsize=size, interval='1min')
        self.daily_data, self.daily_meta = av_ts.get_daily_adjusted(symbol=self.ticker, outputsize=size)
        self.current, _ = av_ts.get_quote_endpoint(symbol=self.ticker)

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
    def help():
        print("Input 'help' for any argument to see input options.")

    @staticmethod
    def __get_output_from_list(choice, options):
        output = 'Please choose one of the following options by number: \n'
        if type(options) == list:
            if choice in options:
                return choice
            else:
                if choice.lower() != 'help':
                    print('"' + choice + '" is not a valid option.')
                for i, o in enumerate(options):
                    output = output + str(i + 1) + '. ' + o + '\n'
                index = int(input(output)) - 1
                return options[index]
        elif type(options) == dict:
            if choice in options.keys():
                return options[choice]
            elif choice in options.values():
                return choice
            else:
                if choice.lower() != 'help':
                    print('"' + choice + '" is not a valid option.')
                for key, val in options.items():
                    if str(val + 1) in output:
                        output = output[:-1] + ' (' + key + ')\n'
                    else:
                        output = output + str(val + 1) + '. ' + key + '\n'
                index = int(input(output)) - 1
                return index

    @staticmethod
    def __get_output_int(choice, minimum=np.inf, maximum=np.inf):
        if type(choice) != int or choice < minimum or choice > maximum:
            choice = input(
                'Please select an integer between {} and {}, {} is not a valid choice:\n'.format(str(minimum - 1),
                                                                                                 str(maximum),
                                                                                                 str(choice)))
        return int(choice)

    @Alias('sma', 'SMA')
    def simple_moving_average(self, interval='daily', series_type='close'):
        """
        :param interval: a string defining the period of each average the options are:
                        ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        :param series_type: a string defining which values (for 'daily', 'weekly', 'monthly') to use.
                    The options are the following: ['close', 'open', 'high', 'low']
        :return: returns a dataframe with a datetime index and floats for values
        """
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        series_options = ['close', 'open', 'high', 'low']
        interval = self.__get_output_from_list(interval, interval_options)
        series_type = self.__get_output_from_list(series_type, series_options)
        return av_ti.get_sma(symbol=self.ticker, interval=interval, time_period=200, series_type=series_type)

    @Alias('ema', 'EMA')
    def exponential_moving_average(self, interval='daily', series_type='close'):
        """
        :param interval: a string defining the period of each average the options are:
                        ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        :param series_type: a string defining which values (for 'daily', 'weekly', 'monthly') to use.
                    The options are the following: ['close', 'open', 'high', 'low']
        :return: returns a dataframe with a datetime index and floats for values
        """
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        series_options = ['close', 'open', 'high', 'low']
        interval = self.__get_output_from_list(interval, interval_options)
        series_type = self.__get_output_from_list(series_type, series_options)
        return av_ti.get_ema(symbol=self.ticker, interval=interval, time_period=200, series_type=series_type)

    # TODO: WMA

    # TODO: DEMA

    # TODO: TEMA

    # TODO: TRIMA

    # TODO: KAMA

    # TODO: MAMA

    @Alias('vwap', 'VWAP')
    def volume_weighted_average_price(self, interval='15min'):
        interval_options = ['1min', '5min', '15min', '30min', '60min']
        interval = self.__get_output_from_list(interval, interval_options)
        return av_ti.get_vwap(symbol=self.ticker, interval=interval)

    # TODO: T3

    @Alias('macd', 'MACD')
    def moving_average_convergence_divergence(self, interval='daily', series_type='close', fastperiod=12,
                                              slowperiod=26, signalperiod=9):
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        series_options = ['close', 'open', 'high', 'low']
        interval = self.__get_output_from_list(interval, interval_options)
        series_type = self.__get_output_from_list(series_type, series_options)
        fastperiod = self.__get_output_int(fastperiod, 1)
        slowperiod = self.__get_output_int(slowperiod, 1)
        signalperiod = self.__get_output_int(signalperiod, 1)
        return av_ti.get_macd(symbol=self.ticker, interval=interval, fastperiod=fastperiod,
                              slowperiod=slowperiod, signalperiod=signalperiod, series_type=series_type)
    # TODO: MACDEXT

    @staticmethod
    def __check_convert(entry, options):
        if type(entry) in (int, float):
            return int(entry)
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
                return Stock.__get_output_from_list(entry, options)

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
        fast_k_period = self.__get_output_int(fast_k_period, 1)
        slow_k_period = self.__get_output_int(slow_k_period, 1)
        slow_d_period = self.__get_output_int(slow_d_period, 1)
        interval = self.__get_output_from_list(interval, interval_options)
        return av_ti.get_stoch(self.ticker, interval=interval, fastkperiod=fast_k_period, slowkperiod=slow_k_period,
                               slowdperiod=slow_d_period, slowkmatype=slow_k_ma_type, slowdmatype=slow_d_ma_type)

    # TODO: STOCHF

    @Alias('rsi', 'RSI', 'relative_strength')
    def relative_strength_index(self, interval='daily', time_period=200, series_type='close'):
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        series_options = ['close', 'open', 'high', 'low']
        interval = self.__get_output_from_list(interval, interval_options)
        series_type = self.__get_output_from_list(series_type, series_options)
        time_period = self.__get_output_int(time_period, 1)
        return av_ti.get_rsi(self.ticker, interval=interval, time_period=time_period, series_type=series_type)

    # TODO: STOCHRSI

    # TODO: WILLR

    @Alias('adx', 'ADX', 'average_directional_movement')
    def average_directional_movement_index(self, interval='daily', time_period=200):
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        interval = self.__get_output_from_list(interval, interval_options)
        time_period = self.__get_output_int(time_period, 1)
        return av_ti.get_adx(symbol=self.ticker, interval=interval, time_period=time_period)

    # TODO: ADXR

    # TODO: APO

    # TODO: PPO

    # TODO: MOM

    # TODO: BOP

    @Alias('cci', 'CCI', 'commodity_channel')
    def commodity_channel_index(self, interval='daily', time_period=200):
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        interval = self.__get_output_from_list(interval, interval_options)
        time_period = self.__get_output_int(time_period, 1)
        return av_ti.get_cci(symbol=self.ticker, interval=interval, time_period=time_period)

    # TODO: CMO

    # TODO: ROC

    # TODO: ROCR

    @Alias('Aroon', 'AROON')
    def aroon(self, interval='daily', time_period=200):
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        interval = self.__get_output_from_list(interval, interval_options)
        time_period = self.__get_output_int(time_period, 1)
        return av_ti.get_aroon(symbol=self.ticker, interval=interval, time_period=time_period)

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
    def bollinger_bands(self, interval='daily', time_period=200, series_type='close',
                        nbdevup=2, nbdevdn=2, matype=0):
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        series_options = ['close', 'open', 'high', 'low']
        type_options = {
            'Simple Moving Average': 0, 'SMA': 0, 'Exponential Moving Average': 1, 'EMA': 1,
            'Weighted Moving Average': 2, 'WMA': 2, 'Double Exponential Moving Average': 3, 'DEMA': 3,
            'Triple Exponential Moving Average': 4, 'TEMA': 4, 'Triangular Moving Average': 5, 'TRIMA': 5,
            'T3 Moving Average': 6, 'Kaufman Adaptive Moving Average': 7, 'KAMA': 7,
            'MESA Adaptive Moving Average': 8, 'MAMA': 8
        }
        interval = self.__get_output_from_list(interval, interval_options)
        series_type = self.__get_output_from_list(series_type, series_options)
        matype = self.__get_output_from_list(matype, type_options)
        time_period = self.__get_output_int(time_period, 1)
        nbdevup = self.__get_output_int(nbdevup, 1)
        nbdevdn = self.__get_output_int(nbdevdn, 1)
        return av_ti.get_bbands(symbol=self.ticker, interval=interval, series_type=series_type, time_period=time_period,
                                nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    # TODO: MIDPOINT

    # TODO: MIDPRICE

    # TODO: SAR

    # TODO: TRANGE

    # TODO: ATR

    # TODO: NATR

    @Alias('ad', 'AD', 'Chaikin_AD_Line', 'Chaikin_AD_line', 'chaikin_ad_line')
    def chaikin_ad_line_values(self, interval='daily'):
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        interval = self.__get_output_from_list(interval, interval_options)
        return av_ti.get_ad(symbol=self.ticker, interval=interval)

    # TODO: ADOSC

    # TODO: OBV*
    @Alias('obv', 'OBV', 'balance_volume')
    def on_balance_volume(self, interval='daily'):
        interval_options = ['1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly']
        interval = self.__get_output_from_list(interval, interval_options)
        return av_ti.get_obv(symbol=self.ticker, interval=interval)

    # TODO: HT_TRENDLINE

    # TODO: HT_SINE

    # TODO: HT_TRENDMODE

    # TODO: HT_DCPERIOD

    # TODO: HT_DCPHASE

    # TODO: HT_PHASOR

        pass
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
    print(s.current)

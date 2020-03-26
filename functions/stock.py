from datetime import timedelta

import pandas as pd
import yfinance
from talib import abstract as ta
from alpha_vantage.timeseries import TimeSeries
from dateutil.relativedelta import relativedelta

from alias import *
from errors import *
from stock_helpers import *

alpha_vantage_KEY = '7ZDI2M6PEWCEOSFC'
av_ts = TimeSeries(alpha_vantage_KEY)


def ticker_search(name):
    return av_ts.get_symbol_search(name)


class Stock:

    def __init__(self, ticker=None, name=None, verbose=True):
        self.ticker = ticker
        self.name = name
        self.__set_name_ticker()
        self.__obj = yfinance.Ticker(self.ticker)
        self.__gen_info = self.__obj.info
        self.summary = self.__gen_info['longBusinessSummary']
        self.sector = self.__gen_info['sector']
        self.industry = self.__gen_info['industry']
        self.bid = self.__gen_info['bid']
        self.ask = self.__gen_info['ask']
        if verbose:
            self.dividend_rate = self.__gen_info['dividendRate']
            self.beta = self.__gen_info['beta']
            self.trailing_PE = self.__gen_info['trailingPE']
            self.market_cap = self.__gen_info['marketCap']
            self.price_to_sales_12m = self.__gen_info['priceToSalesTrailing12Months']
            self.forward_PE = self.__gen_info['forwardPE']
            self.tradeable = self.__gen_info['tradeable']
            self.dividend_yield = self.__gen_info['dividendYield']
            self.forward_EPS = self.__gen_info['forwardEps']
            self.profit_margin = self.__gen_info['profitMargins']
            self.trailing_EPS = self.__gen_info['trailingEps']
            self.actions = self.__obj.actions
            self.dividends = self.__obj.dividends
            self.splits = self.__obj.splits
            self.financials = self.__obj.financials
            self.quarterly_financials = self.__obj.quarterly_financials
            self.major_holders = self.__obj.major_holders
            self.institutional_holders = self.__obj.institutional_holders
            self.balance_sheet = self.__obj.balance_sheet
            self.quarterly_balance_sheet = self.__obj.quarterly_balance_sheet
            self.cashflow = self.__obj.cashflow
            self.quarterly_cashflow = self.__obj.quarterly_cashflow
            self.sustainability = self.__obj.sustainability
            self.recommendations = self.__obj.recommendations
            self.next_event = self.__obj.calendar
            try:
                self.option_expirations = self.__obj.options
            except IndexError:
                self.option_expirations = None

    def __str__(self):
        return self.ticker

    def get_calls(self, dt):
        dt = check_convert_date(dt, 'option expiration date')
        dt = dt.strftime('%Y-%m-%d')
        dt = check_list_options(dt, self.option_expirations, 'option expiration date')
        opt = self.__obj.option_chain(dt)
        return opt.calls

    def get_puts(self, dt):
        dt = check_convert_date(dt, 'option expiration date')
        dt = dt.strftime('%Y-%m-%d')
        dt = check_list_options(dt, self.option_expirations, 'option expiration date')
        opt = self.__obj.option_chain(dt)
        return opt.puts

    def __set_name_ticker(self):
        if self.ticker is None and self.name is None:
            raise NoTickerError
        elif self.ticker is not None:
            vals = ticker_search(self.ticker)[0]
        else:
            vals = ticker_search(self.name)[0]
        if len(vals) == 0:
            print('No stocks found, please try again with a new ticker/name.')
        elif float(vals[0]['9. matchScore']) > 0.7:
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

    @property
    def _get_info(self):
        return self.__gen_info


@aliased
class HistoricalStock(Stock):

    def __init__(self, ticker=None, name=None, start=None, end=None, period=None,
                 interval='1d', adjusted=False, prepost=False, verbose=True):
        super().__init__(ticker, name, verbose)
        self.__dates_bool = False if start is None and end is None else True
        self.__period_bool = False if period is None else True
        self.adjusted = adjusted
        self.prepost = prepost
        if self.__dates_bool:
            self.start = check_convert_date(start, 'start')
            self.end = check_convert_date(end, 'end')
        elif self.__period_bool:
            self.__period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
            self.period = check_list_options(period, self.__period_options, 'period')
        self.__interval_options = ['1m', '2m', '5m', '15m', '30m', '60m',
                                   '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        self.interval = check_list_options(interval, self.__interval_options, 'interval')
        if self.__period_bool:
            if self.interval == '1m' and self.__period_options.index(self.period) >= 2:
                end = date.today()
                start = date.today() - relativedelta(months=1)
                next_date = date.today() + timedelta(days=7)
                self.__hist_info = pd.DataFrame()
                while next_date < end:
                    print(start, next_date)
                    holder = yfinance.download(self.ticker, start=start, end=min(end, next_date),
                                               interval=self.interval, auto_adjust=self.adjusted, prepost=self.prepost,
                                               threads=True)
                    start = next_date
                    next_date = start + timedelta(days=7)
                    self.__hist_info = self.__hist_info.append(holder)
            else:
                self.__hist_info = yfinance.download(self.ticker, period=self.period, interval=self.interval,
                                                     auto_adjust=self.adjusted, prepost=self.prepost, threads=True)
        elif self.__dates_bool:
            if self.interval == '1m' and (self.end - self.start).days > 7:
                start = self.start
                next_date = self.start - timedelta(days=7)
                self.__hist_info = pd.DataFrame()
                while next_date > self.end:
                    holder = yfinance.download(self.ticker, start=start, end=min(end, next_date),
                                               interval=self.interval, auto_adjust=self.adjusted, prepost=self.prepost,
                                               threads=True)
                    start = next_date
                    next_date = start - timedelta(days=7)
                    self.__hist_info = self.__hist_info.append(holder)
            else:
                self.__hist_info = yfinance.download(self.ticker, start=self.start, end=self.end,
                                                     interval=self.interval, auto_adjust=self.adjusted,
                                                     prepost=self.prepost, threads=True)
        self.__hist_info.columns = self.__hist_info.columns.str.lower()
        self.__hist_info.volume = self.__hist_info.volume.astype(float)

    def __str__(self):
        if self.__dates_bool:
            return self.ticker + f' dates: {self.start}-{self.end} ({self.interval})'
        else:
            return self.ticker + f' period: {self.period} ({self.interval})'

    @Alias('sma', 'SMA')
    def simple_moving_average(self, num_periods=3, series_type='close'):
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_options = ['close', 'open', 'high', 'low']
        series_type = check_list_options(series_type, series_options, 'series type')
        return ta.SMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('ema', 'EMA')
    def exponential_moving_average(self, num_periods=3, series_type='close'):
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_options = ['close', 'open', 'high', 'low']
        series_type = check_list_options(series_type, series_options, 'series type')
        return ta.EMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('vwap', 'VWAP')
    def volume_weighted_average_price(self):
        cols = ['High', 'Low', 'Close']
        typical_price = self.__hist_info.loc[:, cols].sum(axis=1).div(3)
        vwap = typical_price * self.__hist_info.loc[:, 'Volume']
        return pd.Series(vwap.values, self.__hist_info.index, name='VWAP')

    @Alias('wma', 'WMA')
    def weighted_moving_average(self, num_periods=10, series_type='close'):
        return ta.WMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('dema', 'DEMA')
    def double_exponential_moving_average(self, num_periods=10, series_type='close'):
        return ta.DEMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('tema', 'TEMA', 't3', 'T3')
    def triple_exponential_moving_average(self, num_periods=5, vfactor=0.7, series_type='close'):
        return ta.TEMA(self.__hist_info, timeperiod=num_periods, vfactor=vfactor, price=series_type)

    @Alias('trima', 'TRIMA')
    def triangular_moving_average(self, num_periods=10, series_type='close'):
        return ta.TRIMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('kama', 'KAMA')
    def kaufman_adaptive_moving_average(self, num_periods=10, series_type='close'):
        return ta.KAMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('mama', 'MAMA', 'mesa', 'MESA')
    def mesa_adaptive_moving_average(self, fast=0, slow=0, series_type='close'):
        return ta.MAMA(self.__hist_info, fastlimit=fast, slowlimit=slow, price=series_type)

    @Alias('macd', 'MACD')
    def moving_average_convergence_divergence(self, slow=26, fast=12, signal=9, series_type='close'):
        assert fast > 0, 'Short period must be greater than 0'
        assert slow > fast, 'Long period must be greater than 0'
        series_options = ['close', 'open', 'high', 'low']
        series_type = check_list_options(series_type, series_options, 'series type')
        return ta.MACD(self.__hist_info, price=series_type, fastperiod=fast, slowperiod=slow, signal_period=signal)

    @Alias('macdext', 'MACDEXT')
    def moving_average_convergence_divergence_matype(self, slow=26, slow_matype=0, fast=12, fast_matype=0, signal=9,
                                                     signal_matype=0, series_type='close'):
        slow_matype = check_matype(slow_matype, 'slow_matype')
        fast_matype = check_matype(fast_matype, 'fast_matype')
        signal_matype = check_matype(signal_matype, 'signal_matype')
        return ta.MACDEXT(self.__hist_info, price=series_type, fastperiod=fast, fastmatype=fast_matype, slowperiod=slow,
                          slowmatype=slow_matype, signalperiod=signal, signalmatype=signal_matype)

    @Alias('stoch', 'STOCH', 'stoch_oscillator')
    def stochastic_oscillator(self, fast_k_period=5, slow_k_period=3, slow_d_period=3,
                              slow_k_ma_type=0, slow_d_ma_type=0):
        slow_k_ma_type = check_matype(slow_k_ma_type, 'slow_k_ma_type')
        slow_d_ma_type = check_matype(slow_d_ma_type, 'slow_d_ma_type')
        return ta.STOCH(self.__hist_info, fastk_period=fast_k_period, slowk_period=slow_k_period,
                        slowk_matype=slow_k_ma_type, slowd_period=slow_d_period, slowd_matype=slow_d_ma_type)

    @Alias('stochf', 'STOCHF')
    def stochastic_fast(self, fast_k_period=5, fast_d_period=3, matype=0):
        matype = check_matype(matype, 'matype')
        return ta.STOCHF(self.__hist_info, fastk_period=fast_k_period, fastd_period=fast_d_period, fastd_matype=matype)

    @Alias('stochrsi', 'STOCHRSI')
    def stochastic_relative_strength_index(self, num_periods=14, series_type='close', fast_k_period=5, fast_d_period=3,
                                           matype=0):
        matype = check_matype(matype, 'matype')
        return ta.STOCHRSI(self.__hist_info, timeperiod=num_periods, price=series_type, fastk_period=fast_k_period,
                           fastd_period=fast_d_period, fastd_matype=matype)

    @Alias('rsi', 'RSI', 'relative_strength')
    def relative_strength_index(self, num_periods=5, series_type='close'):
        return ta.RSI(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('willr', 'WILLR')
    def williams_r(self, num_periods=14):
        return ta.WILLR(self.__hist_info, timeperiod=num_periods)

    @Alias('atr', 'ATR')
    def average_true_range(self, num_periods=14):
        return ta.ATR(self.__hist_info, timeperiod=num_periods)

    @Alias('adx', 'ADX', 'average_directional_movement')
    def average_directional_movement_index(self, num_periods=14):
        return ta.ADX(self.__hist_info, timeperiod=num_periods)

    @Alias('adxr', 'ADXR')
    def average_directional_movement_index_rating(self, num_periods=14):
        return ta.ADXR(self.__hist_info, timeperiod=num_periods)

    @Alias('apo', 'APO')
    def absolute_price_oscillator(self, series_type='close', fast=12, slow=26, matype=0):
        return ta.APO(self.__hist_info, price=series_type, fastperiod=fast, slowperiod=slow, matype=matype)

    @Alias('ppo', 'PPO')
    def percentage_price_oscillator(self, series_type='close', fast=12, slow=26, matype=0):
        matype = check_matype(matype, 'matype')
        return ta.PPO(self.__hist_info, price=series_type, fastperiod=fast, slowperiod=slow, matype=matype)

    @Alias('mom', 'MOM')
    def momentum(self, num_periods=10, series_type='close'):
        return ta.MOM(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('bop', 'BOP')
    def balance_of_power(self):
        return ta.BOP(self.__hist_info)

    @Alias('cci', 'CCI', 'commodity_channel')
    def commodity_channel_index(self, num_periods=20):
        return ta.CCI(self.__hist_info, timeperiod=num_periods)

    @Alias('cmo', 'CMO')
    def chande_momentum_oscillator(self, num_periods=14, series_type='close'):
        return ta.CMO(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('roc', 'ROC')
    def rate_of_change(self, num_periods=10, series_type='close'):
        return ta.ROC(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('rocr', 'ROCR')
    def rate_of_change_ratio(self, num_periods=10, series_type='close'):
        return ta.ROCR(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('Aroon', 'AROON')
    def aroon(self, num_periods=14):
        return ta.AROON(self.__hist_info, timeperiod=num_periods)

    @Alias('Aroonosc', 'AROONOSC', 'AroonOSC', 'AroonOsc')
    def aroon_oscillator(self, num_periods=14):
        return ta.AROONOSC(self.__hist_info, timeperiod=num_periods)

    @Alias('mfi', 'MFI')
    def money_flow_index(self, num_periods=14):
        return ta.MFI(self.__hist_info, timeperiod=num_periods)

    @Alias('TRIX', '1ROC_TEMA', '1ROC_T3')
    def trix(self, num_periods=10, series_type='close'):
        return ta.TRIX(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('ultosc', 'ULTOSC')
    def ultimate_oscillator(self, num_periods1=7, num_periods2=14, num_periods3=28):
        return ta.ULTOSC(self.__hist_info, timeperiod1=num_periods1, timeperiod2=num_periods2, timeperiod3=num_periods3)

    @Alias('dx', 'DX')
    def directional_movement_index(self, num_periods=14):
        return ta.DX(self.__hist_info, timeperiod=num_periods)

    @Alias('minus_di', 'MINUS_DI')
    def minus_directional_indicator(self, num_periods=14):
        return ta.MINUS_DI(self.__hist_info, timeperiod=num_periods)

    @Alias('plus_di', 'PLUS_DI')
    def plus_directional_indicator(self, num_periods=14):
        return ta.PLUS_DI(self.__hist_info, timeperiod=num_periods)

    @Alias('minus_dm', 'MINUS_DM')
    def minus_directional_movement(self, num_periods=14):
        return ta.MINUS_DM(self.__hist_info, timeperiod=num_periods)

    @Alias('plus_dm', 'PLUS_DM')
    def plus_directional_movement(self, num_periods=14):
        return ta.PLUS_DM(self.__hist_info, timeperiod=num_periods)

    @Alias('bbands', 'BBANDS', 'Bollinger_bands')
    def bollinger_bands(self, num_periods=5, dev_up=2, dev_dw=2, matype=0):
        assert dev_up > 0, 'dev_up must be greater than zero'
        assert dev_dw > 0, 'dev_dw must be greater than zero'
        matype = check_matype(matype, 'matype')
        return ta.BBANDS(self.__hist_info, timeperiod=num_periods, nbdevup=dev_up, nbdevdn=dev_dw, matype=matype)

    @Alias('MIDPOINT')
    def midpoint(self, num_periods=14, series_type='close'):
        return ta.MIDPOINT(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('MIDPRICE')
    def midprice(self, num_periods=14):
        return ta.MIDPRICE(self.__hist_info, timeperiod=num_periods)

    @Alias('sar', 'SAR')
    def parabolic_sar(self, acceleration=0, maximum=0):
        return ta.SAR(self.__hist_info, acceleration=acceleration, maximum=maximum)

    @Alias('trange', 'TRANGE', 'TRange')
    def true_range(self):
        return ta.TRANGE(self.__hist_info)

    @Alias('atr', 'ATR', 'AvgTRANGE', 'AvgTRange')
    def average_true_range(self, num_periods=14):
        return ta.ATR(self.__hist_info, timeperiod=num_periods)

    @Alias('natr', 'NATR')
    def normalized_average_true_range(self, num_periods=14):
        return ta.NATR(self.__hist_info, timeperiod=num_periods)

    @Alias('ad', 'AD', 'Chaikin_AD_Line', 'Chaikin_AD_line', 'chaikin_ad_line')
    def chaikin_ad_line_values(self):
        return ta.AD(self.__hist_info)

    @Alias('adosc', 'ADOSC', 'Chaikin_AD_Oscillator')
    def chaikin_ad_oscillator(self, fast=3, slow=10):
        return ta.ADOSC(self.__hist_info, fastperiod=fast, slowperiod=slow)

    @Alias('obv', 'OBV', 'balance_volume')
    def on_balance_volume(self, num_periods=5):
        return ta.OBV(self.__hist_info)

    # TODO: HT_TRENDLINE

    # TODO: HT_SINE

    # TODO: HT_TRENDMODE

    # TODO: HT_DCPERIOD

    # TODO: HT_DCPHASE

    # TODO: HT_PHASOR

    @property
    def get_hist(self):
        return self.__hist_info


if __name__ == '__main__':
    s = HistoricalStock('MSFT', period='1mo', interval='1d')
    print(s.triple_exponential_moving_average())

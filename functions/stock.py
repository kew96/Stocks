from alpha_vantage.timeseries import TimeSeries
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from errors import *
from alias import *
from helper_functions.stock_helpers import *
import pandas as pd
import yfinance

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

    def __str__(self):
        if self.__dates_bool:
            return self.ticker + f' dates: {self.start}-{self.end} ({self.interval})'
        else:
            return self.ticker + f' period: {self.period} ({self.interval})'

    @Alias('sma', 'SMA')
    def simple_moving_average(self, num_periods=3, series_type='Close', other=None):
        assert num_periods > 0, 'num_periods must be greater than 0'
        if other is not None:
            return other.rolling(window=num_periods).mean()
        else:
            series_options = ['Close', 'Open', 'High', 'Low']
            series_type = check_list_options(series_type, series_options, 'series type')
            return self.__hist_info.loc[:, series_type].rolling(window=num_periods).mean().iloc[num_periods - 1:]

    @Alias('ema', 'EMA')
    def exponential_moving_average(self, num_periods=3, series_type='Close', other=None):
        assert num_periods > 0, 'num_periods must be greater than 0'
        if other is not None:
            return other.ewm(span=num_periods, adjust=False).mean()
        else:
            series_options = ['Close', 'Open', 'High', 'Low']
            series_type = check_list_options(series_type, series_options, 'series type')
            return self.__hist_info.loc[:, series_type].ewm(span=num_periods, adjust=False).mean().iloc[
                   num_periods - 1:]

    @Alias('vwap', 'VWAP')
    def volume_weighted_average_price(self):
        cols = ['High', 'Low', 'Close']
        typical_price = self.__hist_info.loc[:, cols].sum(axis=1).div(3)
        vwap = typical_price * self.__hist_info.loc[:, 'Volume']
        return pd.Series(vwap.values, self.__hist_info.index, name='VWAP')

    # TODO: WMA

    # TODO: DEMA

    # TODO: TEMA

    # TODO: TRIMA

    # TODO: KAMA

    # TODO: MAMA

    # TODO: T3

    @Alias('macd', 'MACD')
    def moving_average_convergence_divergence(self, long=26, short=12, series_type='Close'):
        assert short > 0, 'Short period must be greater than 0'
        assert long > short, 'Long period must be greater than 0'
        series_options = ['Close', 'Open', 'High', 'Low']
        series_type = check_list_options(series_type, series_options, 'series type')
        long_ema = self.exponential_moving_average(long, series_type)
        short_ema = self.exponential_moving_average(short, series_type)
        if len(long_ema) == 0 or len(short_ema) == 0:
            assert NoDataError
        return short_ema - long_ema

    # TODO: MACDEXT

    @Alias('stoch', 'STOCH', 'stoch_oscillator')
    def stochastic_oscillator(self, fast_k_period=5, slow_k_period=3, slow_d_period=3,
                              slow_k_ma_type=0, d_ma_type=0, k_args=(), d_args=()):
        type_options = [self.simple_moving_average, self.exponential_moving_average]
        slow_k_ma_type = type_options[slow_k_ma_type]
        d_ma_type = type_options[d_ma_type]
        slow_k_lows = self.__hist_info.loc[:, 'Low'].rolling(window=slow_k_period).min()
        slow_k_highs = self.__hist_info.loc[:, 'High'].rolling(window=slow_k_period).max()
        fast_k_lows = self.__hist_info.loc[:, 'Low'].rolling(window=fast_k_period).min()
        fast_k_highs = self.__hist_info.loc[:, 'High'].rolling(window=fast_k_period).max()
        slow_k = self.__hist_info.loc[:, 'Close'].subtract(slow_k_lows, axis=0).div(slow_k_highs.subtract(
            slow_k_lows, axis=0), axis=0)
        print(type(slow_k))
        slow_k = slow_k_ma_type(other=slow_k, *k_args)
        slow_d = d_ma_type(other=slow_k, num_periods=slow_d_period, *d_args)
        fast_k = self.__hist_info.loc[:, 'Close'].subtract(fast_k_lows, axis=0).div(fast_k_highs.subtract(
            fast_k_lows, axis=0), axis=0)
        fast_d = d_ma_type(other=fast_k, *d_args)
        total = pd.DataFrame({'fast_d': fast_d, 'slow_d': slow_d, 'fast_k': fast_k, 'slow_k': slow_k})
        return total.dropna(how='all')

    # TODO: STOCHF

    @Alias('rsi', 'RSI', 'relative_strength')
    def relative_strength_index(self, num_periods=5, series_type='Close', avg_func=0):
        type_options = [self.simple_moving_average, self.exponential_moving_average]
        avg_func = type_options[avg_func]
        price_dif = self.__hist_info.loc[:, series_type].diff()
        days_up, days_down = price_dif.copy(), price_dif.copy()
        days_up[days_up < 0] = 0
        days_down[days_down > 0] = 0
        up_avg = avg_func(other=days_up, num_periods=num_periods)
        down_avg = avg_func(other=days_down.abs(), num_periods=num_periods)
        return (100 * up_avg / (up_avg + down_avg)).dropna()

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

    @Alias('ad', 'AD', 'Chaikin_AD_Line', 'Chaikin_AD_line', 'chaikin_ad_line')
    def chaikin_ad_line_values(self):  # TODO: implement
        pass

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

    @property
    def get_hist(self):
        return self.__hist_info


if __name__ == '__main__':
    s = HistoricalStock('MSFT', period='1mo', interval='1d')
    # print(type(s.get_hist))
    print(s.relative_strength_index())

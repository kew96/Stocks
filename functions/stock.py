from datetime import timedelta

import pandas as pd
import yfinance
from alpha_vantage.timeseries import TimeSeries
from dateutil.relativedelta import relativedelta
from talib import abstract as ta

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
    def simple_moving_average(self, num_periods: int = 3, series_type: str = 'close') -> pd.Series:
        """
        The arithmetic average over the past n time periods.


        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the simple moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.SMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('ema', 'EMA')
    def exponential_moving_average(self, num_periods: int = 3, series_type: str = 'close') -> pd.Series:
        """
        The geometric average over the past n time periods

        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the exponential moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.EMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('vwap', 'VWAP')
    def volume_weighted_average_price(self) -> pd.Series:
        """
        The daily price times volume summed divided by the total volume of the period

        :return: volume weighted average price for the entire time period of historical data
        :rtype: pandas.Series
        """
        cols = ['High', 'Low', 'Close']
        typical_price = self.__hist_info.loc[:, cols].sum(axis=1).div(3)
        vwap = typical_price * self.__hist_info.loc[:, 'Volume']
        return pd.Series(vwap.values, self.__hist_info.index, name='VWAP')

    @Alias('wma', 'WMA')
    def weighted_moving_average(self, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        Calculates the average of the time frame for the specified series type with greater weights on more recent data

        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the weighted moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.WMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('dema', 'DEMA')
    def double_exponential_moving_average(self, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The double exponential moving average. Similar to exponential moving average but is more smoothed. Think of
        a moving average of a moving average.

        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the double exponential moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.DEMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('tema', 'TEMA', 't3', 'T3')
    def triple_exponential_moving_average(self, num_periods: int = 5, vfactor: float = 0.7,
                                          series_type: str = 'close') -> pd.Series:
        """
        Based on DEMA, this smooths the data with multiple exponential moving averages and scaling factors.

        :param num_periods: specifies the number of periods for each calculation
        :param vfactor: a value that indicates the smoothing of the function, between 0 and 1. If 1, then TEMA = DEMA
        while if 0, TEMA = EMA
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the triple exponential moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        assert 0. <= vfactor <= 1., 'vfactor must be between 0 and 1 inclusive'
        series_type = check_series_type(series_type)
        return ta.TEMA(self.__hist_info, timeperiod=num_periods, vfactor=vfactor, price=series_type)

    @Alias('trima', 'TRIMA')
    def triangular_moving_average(self, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        A form of weighted moving average that puts more weight on the middle of the time frame and less on the ends.

        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the triangular moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.TRIMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('kama', 'KAMA')
    def kaufman_adaptive_moving_average(self, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        A moving average designed to account for market noise/volatility

        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the kaufman adaptive moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.KAMA(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('mama', 'MAMA', 'mesa', 'MESA')
    def mesa_adaptive_moving_average(self, fast: float = 0.01, slow: float = 0.01,
                                     series_type: str = 'close') -> pd.DataFrame:
        """
        Trend following indicator based on the rate change of phase as measured by the Hilbert Transform Discriminator.
        Uses a fast and slow moving average to quickly respond to price changes.

        :param fast: specifies the number of periods for each fast moving average calculation
        :param slow: specifies the number of periods for each slow moving average calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas DataFrame with the mesa adaptive moving average: first column is "mama",
        second column is "fama"
        :rtype: pandas.DataFrame
        """
        assert fast >= 0, 'fast must be greater than or equal to 0'
        assert slow >= 0, 'slow must be greater than or equal to 0'
        series_type = check_series_type(series_type)
        return ta.MAMA(self.__hist_info, fastlimit=fast, slowlimit=slow, price=series_type)

    @Alias('macd', 'MACD')
    def moving_average_convergence_divergence(self, slow: int = 26, fast: int = 12, signal: int = 9,
                                              series_type: str = 'close') -> pd.DataFrame:
        """
        The MACD is the difference between two exponential moving averages with a signal line being the exponential
        moving average of the MACD. Signals trend changes and indicates new trends. High values indicate overbought
        conditions, low values indicate oversold conditions. Divergence with the price indicates an end to the current
        trend, especially if the MACD is at extreme high or low values. When the MACD line crosses above the signal
        line a buy signal is generated. When the MACD crosses below the signal line a sell signal is generated. To
        confirm the signal, the MACD should be above zero for a buy, and below zero for a sell.

        :param slow: specifies the number of periods for each slow exponential moving average calculation
        :param fast: specifies the number of periods for each fast exponential moving average calculation
        :param signal: specifies the number of periods for the signal line exponential moving average calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas DataFrame with the moving average convergence/divergence: first column is "macd",
        second
        column is "macdsignal", third column is "macdhist"
        :rtype: pandas.DataFrame
        """
        assert fast > 0, 'Short period must be greater than 0'
        assert slow > fast, 'Long period must be greater than 0'
        assert signal > 0, 'signal must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.MACD(self.__hist_info, price=series_type, fastperiod=fast, slowperiod=slow, signal_period=signal)

    @Alias('macdext', 'MACDEXT')
    def moving_average_convergence_divergence_matype(self, slow: int = 26, slow_matype: int = 0, fast: int = 12,
                                                     fast_matype: int = 0, signal: int = 9, signal_matype: int = 0,
                                                     series_type: str = 'close') -> pd.DataFrame:
        """
        The MACDEXT is the difference between two arbitrary moving averages with a signal line being the arbitrary
        moving average of the MACDEXT. Signals trend changes and indicates new trends. High values indicate overbought
        conditions, low values indicate oversold conditions. Divergence with the price indicates an end to the current
        trend, especially if the MACD is at extreme high or low values. When the MACD line crosses above the signal
        line
        a buy signal is generated. When the MACD crosses below the signal line a sell signal is generated. To confirm
        the signal, the MACDEXT should be above zero for a buy, and below zero for a sell.

        :param slow: specifies the number of periods for each slow arbitrary moving average calculation
        :param slow_matype: matype to use for the slow arbitrary moving average
        :param fast: specifies the number of periods for each fast arbitrary moving average calculation
        :param fast_matype: matype to use for the fast arbitrary moving average
        :param signal: specifies the number of periods for the signal line arbitrary moving average calculation
        :param signal_matype: matype to use for the signal arbitrary moving average
        :param series_type: the price data to calculate over
        :return: returns a pandas DataFrame with the moving average convergence/divergence: first column is "macd",
        second
        column is "macdsignal", third column is "macdhist"
        :rtype: pandas.DataFrame
        """
        assert fast > 0, 'fast must be greater than 0'
        assert slow > fast, 'slow must be greater than fast'
        assert signal > 0, 'signal must be greater than 0'
        slow_matype = check_matype(slow_matype, 'slow_matype')
        fast_matype = check_matype(fast_matype, 'fast_matype')
        signal_matype = check_matype(signal_matype, 'signal_matype')
        series_type = check_series_type(series_type)
        return ta.MACDEXT(self.__hist_info, price=series_type, fastperiod=fast, fastmatype=fast_matype,
                          slowperiod=slow,
                          slowmatype=slow_matype, signalperiod=signal, signalmatype=signal_matype)

    @Alias('stoch', 'STOCH', 'stoch_oscillator')
    def stochastic_oscillator(self, fast_k_period: int = 5, slow_k_period: int = 3, slow_d_period: int = 3,
                              slow_k_ma_type: int = 1, slow_d_ma_type: int = 1) -> pd.DataFrame:
        """
        The Stochastic Oscillator measures where the close is in relation to the recent trading range. The values range
        from zero to 100. %D values over 75 indicate an overbought condition; values under 25 indicate an oversold
        condition. When the Fast %D crosses above the Slow %D, it is a buy signal; when it crosses below, it is a sell
        signal. The Raw %K is generally considered too erratic to use for crossover signals. %K is the original
        calculation and %D takes the moving average of %K, both values are slow which refers to them being smoothed.

        :param fast_k_period: specifies the number of periods for each fast k arbitrary moving average calculation
        :param slow_k_period: specifies the number of periods for each slow k arbitrary moving average calculation
        :param slow_d_period: specifies the number of periods for each slow d arbitrary moving average calculation
        :param slow_k_ma_type: specifies the arbitrary moving average function for each slow k calculation
        :param slow_d_ma_type: specifies the arbitrary moving average function for each slow d calculation
        :return: returns a pandas DataFrame with the slow % values where slow is referring to using the smoothed %K:
        first column is "slowk", second column is "slowd"
        :rtype: pandas.DataFrame
        """
        assert slow_k_period > 0, 'slow_k_period must be greater than 0'
        assert fast_k_period > slow_k_period, 'fast_k_period must be greater than slow_k_period'
        assert slow_d_period > 0, 'slow_d_period must be greater than 0'
        slow_k_ma_type = check_matype(slow_k_ma_type, 'slow_k_ma_type')
        slow_d_ma_type = check_matype(slow_d_ma_type, 'slow_d_ma_type')
        return ta.STOCH(self.__hist_info, fastk_period=fast_k_period, slowk_period=slow_k_period,
                        slowk_matype=slow_k_ma_type, slowd_period=slow_d_period, slowd_matype=slow_d_ma_type)

    @Alias('stochf', 'STOCHF')
    def stochastic_fast(self, fast_k_period: int = 5, fast_d_period: int = 3, matype: int = 1) -> pd.DataFrame:
        """
        The Fast Stochastic Oscillator measures where the close is in relation to the recent trading range. The values
        range from zero to 100. %D values over 75 indicate an overbought condition; values under 25 indicate an
        oversold
        condition. When the Fast %D crosses above the Slow %D, it is a buy signal; when it crosses below, it is a sell
        signal. The Raw %K is generally considered too erratic to use for crossover signals. %K is the original
        calculation and %D takes the moving average of %K, both values are fast which refers to them being un-smoothed.

        :param fast_k_period: specifies the number of periods for each slow k arbitrary moving average calculation
        :param fast_d_period: specifies the number of periods for each fast k arbitrary moving average calculation
        :param matype: specifies the arbitrary moving average function for each fast d calculation
        :return: returns a pandas DataFrame with the fast % values where fast is referring to using the un-smoothed %K:
        first column is "fastk", second column is "fastd"
        :rtype: pandas.DataFrame
        """
        assert fast_k_period > 0, 'fast_k_period must be greater than 0'
        assert fast_d_period > 0, 'fast_d_period must be greater than 0'
        matype = check_matype(matype, 'matype')
        return ta.STOCHF(self.__hist_info, fastk_period=fast_k_period, fastd_period=fast_d_period, fastd_matype=matype)

    @Alias('stochrsi', 'STOCHRSI')
    def stochastic_relative_strength_index(self, num_periods: int = 14, series_type: str = 'close',
                                           fast_k_period: int = 5, fast_d_period: int = 3,
                                           matype: int = 1) -> pd.DataFrame:
        """
        Stochastic RSI (StochRSI) is an indicator of an indicator. It calculates the RSI relative to its range in order
        to increase the sensitivity of the standard RSI. The values of the StochRSI are from zero to one.

        The Stochastic RSI can be interpreted several ways. Overbought/oversold conditions are indicated when the
        StochRSI crosses above .20 / below .80. A buy signal is generated when the StochRSI moves from oversold to
        above
        the midpoint (.50). A sell signal is generated when the StochRSI moves from overbought to below the midpoint.
        Also look for divergence with the price to indicate the end of a trend.

        :param num_periods: The number of periods to calculate over
        :param series_type: The pricing data to use for the calculation
        :param fast_k_period: The number of periods to use for calculating fast (un-smoothed) %K
        :param fast_d_period: The number of periods to use for calculating fast (un-smoothed) %D
        :param matype: The arbitrary function to use as a moving average
        :return: Returns a pandas DataFrame of fast (un-smoothed) values: first column is "fastk", second column is
        "fastd"
        :rtype: pandas.DataFrame
        """
        matype = check_matype(matype, 'matype')
        return ta.STOCHRSI(self.__hist_info, timeperiod=num_periods, price=series_type, fastk_period=fast_k_period,
                           fastd_period=fast_d_period, fastd_matype=matype)

    @Alias('rsi', 'RSI', 'relative_strength')
    def relative_strength_index(self, num_periods: int = 5, series_type: str = 'close') -> pd.Series:
        """
        The Relative Strength Index (RSI) calculates a ratio of the recent upward price movements to the absolute price
        movement. The RSI ranges from 0 to 100. The RSI is interpreted as an overbought/oversold indicator when the
        value is over 70/below 30. You can also look for divergence with price. If the price is making new highs/lows,
        and the RSI is not, it indicates a reversal.

        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.RSI(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('willr', 'WILLR')
    def williams_r(self, num_periods: int = 14) -> pd.Series:
        """
        The Williams %R is similar to an un-smoothed Stochastic %K. The values range from zero to 100, and are charted
        on an inverted scale, that is, with zero at the top and 100 at the bottom. Values below 20 indicate an
        overbought condition and a sell signal is generated when it crosses the 20 line. Values over 80 indicate an
        oversold condition and a buy signal is generated when it crosses the 80 line.

        :param num_periods: The number of periods to perform the rolling calculation over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.WILLR(self.__hist_info, timeperiod=num_periods)

    @Alias('atr', 'ATR')
    def average_true_range(self, num_periods: int = 14) -> pd.Series:
        """
        The ATR is a Welles Wilder style moving average of the True Range. The ATR is a measure of volatility. High ATR
        values indicate high volatility, and low values indicate low volatility, often seen when the price is flat.

        The ATR is a component of the Welles Wilder Directional Movement indicators (+/-DI, DX, ADX and ADXR).

        :param num_periods: The number of periods to use for the rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.ATR(self.__hist_info, timeperiod=num_periods)

    @Alias('adx', 'ADX', 'average_directional_movement')
    def average_directional_movement_index(self, num_periods: int = 14) -> pd.Series:
        """
        The ADX is a Welles Wilder style moving average of the Directional Movement Index (DX). The values range from 0
        to 100, but rarely get above 60. To interpret the ADX, consider a high number to be a strong trend, and a low
        number, a weak trend.

        :param num_periods: The number of periods to perform the rolling calculation over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.ADX(self.__hist_info, timeperiod=num_periods)

    @Alias('adxr', 'ADXR')
    def average_directional_movement_index_rating(self, num_periods: int = 14) -> pd.Series:
        """
        The ADXR is equal to the current ADX plus the ADX from n bars ago divided by 2. In effect, it is the average of
        the two ADX values. The ADXR smooths the ADX, and is therefore less responsive, however, the ADXR filters out
        excessive tops and bottoms. To interpret the ADXR, consider a high number to be a strong trend, and a low
        number, a weak trend.

        :param num_periods: The number of periods to perform the rolling calculation over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.ADXR(self.__hist_info, timeperiod=num_periods)

    @Alias('apo', 'APO')
    def absolute_price_oscillator(self, series_type: str = 'close', fast: int = 12, slow: int = 26,
                                  matype: int = 0) -> pd.Series:
        """
        The Price Oscillator shows the difference between two moving averages. It is basically a MACD, but the Price
        Oscillator can use any time periods. A buy signal is generate when the Price Oscillator rises above zero,
        and a sell signal when the it falls below zero.

        :param series_type: The pricing data to perform the rolling calculation over
        :param fast: The period to perform the fast arbitrary moving average over
        :param slow: The period to perform the slow arbitrary moving average over
        :param matype: The arbitrary moving average to use for calculations
        :return: Returns a pandas Series with calculated values
        :rtype: pandas.Series
        """
        assert fast > 0, 'fast must be greater than 0'
        assert slow > fast, 'slow must be greater than fast'
        series_type = check_series_type(series_type)
        matype = check_matype(matype, 'matype')
        return ta.APO(self.__hist_info, price=series_type, fastperiod=fast, slowperiod=slow, matype=matype)

    @Alias('ppo', 'PPO')
    def percentage_price_oscillator(self, series_type: str = 'close', fast: int = 12, slow: int = 26,
                                    matype: int = 0) -> pd.Series:
        """
        The Price Oscillator Percent shows the percentage difference between two moving averages. A buy signal is
        generate when the Price Oscillator Percent rises above zero, and a sell signal when the it falls below zero.

        :param series_type: The pricing data to perform the rolling calculation over
        :param fast: The period to perform the fast arbitrary moving average over
        :param slow: The period to perform the slow arbitrary moving average over
        :param matype: The arbitrary moving average to use for calculations
        :return: Returns a pandas Series with calculated values
        :rtype: pandas.Series
        """
        assert fast > 0, 'fast must be greater than 0'
        assert slow > fast, 'slow must be greater than fast'
        series_type = check_series_type(series_type)
        matype = check_matype(matype, 'matype')
        return ta.PPO(self.__hist_info, price=series_type, fastperiod=fast, slowperiod=slow, matype=matype)

    @Alias('mom', 'MOM')
    def momentum(self, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The Momentum is a measurement of the acceleration and deceleration of prices. It indicates if prices are 
        increasing at an increasing rate or decreasing at a decreasing rate. The Momentum function can be applied to 
        the price, or to any other data series. 

        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.MOM(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('bop', 'BOP')
    def balance_of_power(self) -> pd.Series:
        """
        The Balance of Power indicator measures the market strength of buyers against sellers by assessing the 
        ability of each side to drive prices to an extreme level. The calculation is: Balance of Power = (Close price –
        Open price) / (High price – Low price) The resulting value can be smoothed by a moving average. 

        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        return ta.BOP(self.__hist_info)

    @Alias('cci', 'CCI', 'commodity_channel')
    def commodity_channel_index(self, num_periods: int = 20) -> pd.Series:
        """
        The CCI is designed to detect beginning and ending market trends. The range of 100 to -100 is the normal
        trading range. CCI values outside of this range indicate overbought or oversold conditions. You can also look
        for price divergence in the CCI. If the price is making new highs, and the CCI is not, then a price
        correction is likely.

        :param num_periods: The number of periods to perform the rolling calculation over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.CCI(self.__hist_info, timeperiod=num_periods)

    @Alias('cmo', 'CMO')
    def chande_momentum_oscillator(self, num_periods: int = 14, series_type: str = 'close') -> pd.Series:
        """
        The Chande Momentum Oscillator is a modified RSI. Where the RSI divides the upward movement by the net
        movement (up / (up + down)), the CMO divides the total movement by the net movement ((up - down) / (up + down)).

        There are several ways to interpret the CMO. Values over 50 indicate overbought conditions, while values
        under -50 indicate oversold conditions. High CMO values indicate strong trends. When the CMO crosses above a
        moving average of the CMO, it is a buy signal, crossing down is a sell signal.

        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.CMO(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('roc', 'ROC')
    def rate_of_change(self, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The Rate of Change function measures rate of change relative to previous periods. The function is used to
        determine how rapidly the data is changing. The factor is usually 100, and is used merely to make the numbers
        easier to interpret or graph. The function can be used to measure the Rate of Change of any data series,
        such as price or another indicator. When used with the price, it is referred to as the Price Rate Of Change,
        or PROC.

        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.ROC(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('rocr', 'ROCR')
    def rate_of_change_ratio(self, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The ratio of the Rate of Change function measures rate of change relative to previous periods. The function is 
        used to determine how rapidly the data is changing. The factor is usually 100, and is used merely to make the 
        numbers easier to interpret or graph. The function can be used to measure the Rate of Change of any data series,
        such as price or another indicator. When used with the price, it is referred to as the Price Rate Of Change,
        or PROC.

        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.ROCR(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('Aroon', 'AROON')
    def aroon(self, num_periods: int = 14) -> pd.DataFrame:
        """
        The word aroon is Sanskrit for "dawn's early light." The Aroon indicator attempts to show when a new trend is
        dawning. The indicator consists of two lines (Up and Down) that measure how long it has been since the
        highest high/lowest low has occurred within an n period range.

        When the Aroon Up is staying between 70 and 100 then it indicates an upward trend. When the Aroon Down is
        staying between 70 and 100 then it indicates an downward trend. A strong upward trend is indicated when the
        Aroon Up is above 70 while the Aroon Down is below 30. Likewise, a strong downward trend is indicated when
        the Aroon Down is above 70 while the Aroon Up is below 30. Also look for crossovers. When the Aroon Down
        crosses above the Aroon Up, it indicates a weakening of the upward trend (and vice versa).

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas DataFrame of calculated values: first column is "aroondown", second column is 
        "aroonup"
        :rtype: pandas.DataFrame
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.AROON(self.__hist_info, timeperiod=num_periods)

    @Alias('Aroonosc', 'AROONOSC', 'AroonOSC', 'AroonOsc')
    def aroon_oscillator(self, num_periods: int = 14) -> pd.Series:
        """
        The Aroon Oscillator is calculated by subtracting the Aroon Down from the Aroon Up. The resultant number will
        oscillate between 100 and -100. The Aroon Oscillator will be high when the Aroon Up is high and the Aroon Down
        is low, indicating a strong upward trend. The Aroon Oscillator will be low when the Aroon Down is high and 
        the Aroon Up is low, indicating a strong downward trend. When the Up and Down are approximately equal, 
        the Aroon Oscillator will hover around zero, indicating a weak trend or consolidation. See the Aroon 
        indicator for more information.

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.AROONOSC(self.__hist_info, timeperiod=num_periods)

    @Alias('mfi', 'MFI')
    def money_flow_index(self, num_periods: int = 14) -> pd.Series:
        """
        The Money Flow Index calculates the ratio of money flowing into and out of a security. To interpret the Money
        Flow Index, look for divergence with price to signal reversals. Money Flow Index values range from 0 to 100. 
        Values above 80/below 20 indicate market tops/bottoms.

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.MFI(self.__hist_info, timeperiod=num_periods)

    @Alias('TRIX', '1ROC_TEMA', '1ROC_T3')
    def trix(self, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The TRIX indicator calculates the rate of change of a triple exponential moving average. The values oscillate
        around zero. Buy/sell signals are generated when the TRIX crosses above/below zero. A (typically) 9 period
        exponential moving average of the TRIX can be used as a signal line. A buy/sell signals are generated when
        the TRIX crosses above/below the signal line and is also above/below zero.


        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.TRIX(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('ultosc', 'ULTOSC')
    def ultimate_oscillator(self, num_periods1: int = 7, num_periods2: int = 14, num_periods3: int = 28) -> pd.Series:
        """
        The Ultimate Oscillator is the weighted sum of three oscillators of different time periods. The typical time
        periods are 7, 14 and 28. The values of the Ultimate Oscillator range from zero to 100. Values over 70
        indicate overbought conditions, and values under 30 indicate oversold conditions. Also look for
        agreement/divergence with the price to confirm a trend or signal the end of a trend.

        :param num_periods1: The number of periods for each of the first rolling calculations
        :param num_periods2: The number of periods for each of the second rolling calculations
        :param num_periods3: The number of periods for each of the third rolling calculations
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods1 > 0, 'num_periods1 must be greater than 0'
        assert num_periods2 > num_periods1, 'num_periods2 must be greater than num_periods1'
        assert num_periods3 > num_periods2, 'num_periods3 must be greater than num_periods2'
        return ta.ULTOSC(self.__hist_info, timeperiod1=num_periods1, timeperiod2=num_periods2,
                         timeperiod3=num_periods3)

    @Alias('dx', 'DX')
    def directional_movement_index(self, num_periods: int = 14) -> pd.Series:
        """
        The DX is usually smoothed with a moving average (i.e. the ADX). The values range from 0 to 100, but rarely
        get above 60. To interpret the DX, consider a high number to be a strong trend, and a low number, a weak trend.

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.DX(self.__hist_info, timeperiod=num_periods)

    @Alias('minus_di', 'MINUS_DI')
    def minus_directional_indicator(self, num_periods: int = 14) -> pd.Series:
        """
        The +DI is the percentage of the true range that is up. The -DI is the percentage of the true range that is
        down. A buy signal is generated when the +DI crosses up over the -DI. A sell signal is generated when the -DI
        crosses up over the +DI. You should wait to enter a trade until the extreme point is reached. That is,
        you should wait to enter a long trade until the price reaches the high of the bar on which the +DI crossed
        over the -DI, and wait to enter a short trade until the price reaches the low of the bar on which the -DI
        crossed over the +DI.

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.MINUS_DI(self.__hist_info, timeperiod=num_periods)

    @Alias('plus_di', 'PLUS_DI')
    def plus_directional_indicator(self, num_periods: int = 14) -> pd.Series:
        """
        The +DI is the percentage of the true range that is up. The -DI is the percentage of the true range
        that is
        down. A buy signal is generated when the +DI crosses up over the -DI. A sell signal is generated when
        the -DI
        crosses up over the +DI. You should wait to enter a trade until the extreme point is reached. That is,
        you should wait to enter a long trade until the price reaches the high of the bar on which the +DI
        crossed
        over the -DI, and wait to enter a short trade until the price reaches the low of the bar on which the
        -DI
        crossed over the +DI.

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.PLUS_DI(self.__hist_info, timeperiod=num_periods)

    @Alias('minus_dm', 'MINUS_DM')
    def minus_directional_movement(self, num_periods: int = 14) -> pd.Series:
        """
        The +DI is the percentage of the true range that is up. The -DI is the percentage of the true range
        that is
        down. A buy signal is generated when the +DI crosses up over the -DI. A sell signal is generated when
        the -DI
        crosses up over the +DI. You should wait to enter a trade until the extreme point is reached. That is,
        you should wait to enter a long trade until the price reaches the high of the bar on which the +DI
        crossed
        over the -DI, and wait to enter a short trade until the price reaches the low of the bar on which the
        -DI
        crossed over the +DI.

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.MINUS_DM(self.__hist_info, timeperiod=num_periods)

    @Alias('plus_dm', 'PLUS_DM')
    def plus_directional_movement(self, num_periods: int = 14) -> pd.Series:
        """
        The +DI is the percentage of the true range that is up. The -DI is the percentage of the true range
        that is
        down. A buy signal is generated when the +DI crosses up over the -DI. A sell signal is generated when
        the -DI
        crosses up over the +DI. You should wait to enter a trade until the extreme point is reached. That is,
        you should wait to enter a long trade until the price reaches the high of the bar on which the +DI
        crossed
        over the -DI, and wait to enter a short trade until the price reaches the low of the bar on which the
        -DI
        crossed over the +DI.

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.PLUS_DM(self.__hist_info, timeperiod=num_periods)

    @Alias('bbands', 'BBANDS', 'Bollinger_bands')
    def bollinger_bands(self, num_periods: int = 5, dev_up: int = 2, dev_dw: int = 2, matype: int = 0) -> pd.DataFrame:
        """
        Bollinger Bands consist of three lines. The middle band is a simple moving average (generally 20 periods) of
        the typical price (TP). The upper and lower bands are F standard deviations (generally 2) above and below the
        middle band. The bands widen and narrow when the volatility of the price is higher or lower, respectively.

        Bollinger Bands do not, in themselves, generate buy or sell signals; they are an indicator of overbought or
        oversold conditions. When the price is near the upper or lower band it indicates that a reversal may be
        imminent. The middle band becomes a support or resistance level. The upper and lower bands can also be
        interpreted as price targets. When the price bounces off of the lower band and crosses the middle band,
        then the upper band becomes the price target.

        :param num_periods: he number of periods for each rolling calculation
        :param dev_up: The standard deviation multiplier for the upper band
        :param dev_dw: The standard deviation multiplier for the lower band
        :param matype: The arbitrary rolling average to use to calculate the middle band
        :return: Returns a pandas DataFrame: first column is "upperband", second column is "middleband", third column
        is "lowerband"
        :rtype: pandas.DataFrame
        """
        assert dev_up > 0, 'dev_up must be greater than 0'
        assert dev_dw > 0, 'dev_dw must be greater than 0'
        assert num_periods > 0, 'num_periods must be greater than 0'
        matype = check_matype(matype, 'matype')
        return ta.BBANDS(self.__hist_info, timeperiod=num_periods, nbdevup=dev_up, nbdevdn=dev_dw, matype=matype)

    @Alias('MIDPOINT')
    def midpoint(self, num_periods: int = 14, series_type: str = 'close') -> pd.Series:
        """
        The rolling Midpoint of the specified pricing data.

        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.MIDPOINT(self.__hist_info, timeperiod=num_periods, price=series_type)

    @Alias('MIDPRICE')
    def midprice(self, num_periods: int = 14) -> pd.Series:
        """
        The rolling Midprice of the specified pricing data calculated by summing the highest high and lowest low and
        then dividing by two.

        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.MIDPRICE(self.__hist_info, timeperiod=num_periods)

    @Alias('sar', 'SAR')
    def parabolic_sar(self, acceleration: float = 0.01, maximum: float = 0.2) -> pd.Series:
        """
        The Parabolic SAR calculates a trailing stop. Simply exit when the price crosses the SAR. The SAR assumes
        that you are always in the market, and calculates the Stop And Reverse point when you would close a long
        position and open a short position or vice versa.

        :param acceleration: A scaling factor to capture momentum
        :param maximum: The lowest or highest point depending on the trend
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert acceleration >= 0, 'acceleration must be greater than or equal to 0'
        assert maximum >= 0, 'maximum must be greater than or equal to 0'
        return ta.SAR(self.__hist_info, acceleration=acceleration, maximum=maximum)

    @Alias('trange', 'TRANGE', 'TRange')
    def true_range(self):  # TODO: complete docstrings
        return ta.TRANGE(self.__hist_info)

    @Alias('atr', 'ATR', 'AvgTRANGE', 'AvgTRange')
    def average_true_range(self, num_periods=14):
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.ATR(self.__hist_info, timeperiod=num_periods)

    @Alias('natr', 'NATR')
    def normalized_average_true_range(self, num_periods=14):
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.NATR(self.__hist_info, timeperiod=num_periods)

    @Alias('ad', 'AD', 'Chaikin_AD_Line', 'Chaikin_AD_line', 'chaikin_ad_line')
    def chaikin_ad_line_values(self):
        return ta.AD(self.__hist_info)

    @Alias('adosc', 'ADOSC', 'Chaikin_AD_Oscillator')
    def chaikin_ad_oscillator(self, fast=3, slow=10):
        assert fast > 0, 'fast must be greater than 0'
        assert slow > fast, 'slow must be greater than fast'
        return ta.ADOSC(self.__hist_info, fastperiod=fast, slowperiod=slow)

    @Alias('obv', 'OBV', 'balance_volume')
    def on_balance_volume(self, num_periods=5):
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.OBV(self.__hist_info, timeperiod=num_periods)

    @Alias('trendline', 'TRENDLINE', 'instantaneous')
    def hilbert_transform_instantaneous_trendline(self, series_type='close'):
        series_type = check_series_type(series_type)
        return ta.HT_TRENDLINE(self.__hist_info, price=series_type)

    @Alias('sine', 'SINE', 'ht_sine', 'HT_SINE', 'sine_wave', 'SINE_WAVE')
    def hilbert_transform_sine_wave(self, series_type='close'):
        series_type = check_series_type(series_type)
        return ta.HT_SINE(self.__hist_info, price=series_type)

    @Alias('trendmode', 'TRENDMODE', 'Trend_vs_Cycle')
    def hilbert_transform_trend_vs_cycle_mode(self, series_type='close'):
        series_type = check_series_type(series_type)
        return ta.HTTRENDMODE(self.__hist_info, price=series_type)

    @Alias('dcperiod', 'DCPERIOD', 'dc_period', 'DC_PERIOD', 'Dominant_Cycle_Period')
    def hilbert_transform_dominant_cycle_period(self, series_type='close'):
        series_type = check_series_type(series_type)
        return ta.HT_DCPERIOD(self.__hist_info, price=series_type)

    @Alias('dcphase', 'DCPHASE', 'dc_phase', 'DC_PHASE', 'Dominant_Cycle_Phase')
    def hilbert_transform_dominant_cycle_phase(self, series_type='close'):
        series_type = check_series_type(series_type)
        return ta.HT_DCPHASE(self.__hist_info, price=series_type)

    @Alias('phasor', 'PHASOR', 'Phasor_Components')
    def hilbert_transform_phasor_components(self, series_type='close'):
        series_type = check_series_type(series_type)
        return ta.HT_PHASOR(self.__hist_info, price=series_type)

    @property
    def get_hist(self):
        return self.__hist_info


if __name__ == '__main__':
    # noinspection SpellCheckingInspection
    s = HistoricalStock('MSFT', period='1mo', interval='1d')
    print(s.bollinger_bands())

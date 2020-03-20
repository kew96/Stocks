from alpha_vantage.timeseries import TimeSeries
from errors import *
from alias import *
from helper_functions.stock_helpers import *
import yfinance

alpha_vantage_KEY = '7ZDI2M6PEWCEOSFC'
av_ts = TimeSeries(alpha_vantage_KEY)


def ticker_search(name):
    return av_ts.get_symbol_search(name)


@aliased
class Stock:

    def __init__(self, ticker=None, name=None, start=None, end=None, interval=None, period=None, verbose=True):
        self.ticker = ticker
        self.name = name
        self.historical = False if start is None and end is None and period is None else True
        self.__dates_bool = False if start is None and end is None else True
        self.__period_bool = False if period is None else True
        if self.__dates_bool:
            self.start = check_convert_date(start, 'start')
            self.end = check_convert_date(end, 'end')
        elif self.__period_bool:
            self.__period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
            self.period = check_list_options(period, self.__period_options, 'period')
        if self.historical:
            self.__interval_options = ['1m', '2m', '5m', '15m', '30m', '60m',
                                       '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
            self.interval = check_list_options(interval, self.__interval_options, 'interval')
        self.__set_name_ticker()
        self.__obj = yfinance.Ticker(self.ticker)
        if self.__period_bool:
            self.__hist_info = self.__obj.history(period=self.period, interval=self.interval)
        elif self.__dates_bool:
            self.__hist_info = self.__obj.history(start=self.start, end=self.end, interval=self.interval)
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
            self.option_expirations = self.__obj.options

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

    @Alias('vwap', 'VWAP')
    def volume_weighted_average_price(self):  # TODO: implement
        pass

    # TODO: WMA

    # TODO: DEMA

    # TODO: TEMA

    # TODO: TRIMA

    # TODO: KAMA

    # TODO: MAMA

    @Alias('macd', 'MACD')
    def moving_average_convergence_divergence(self):  # TODO: implement
        pass

    # TODO: T3

    @Alias('stoch', 'STOCH', 'stoch_oscillator')
    def stochastic_oscillator(self):  # TODO: implement
        pass

    # TODO: MACDEXT

    @Alias('rsi', 'RSI', 'relative_strength')
    def relative_strength_index(self):  # TODO: implement
        pass

    # TODO: STOCHF

    @Alias('adx', 'ADX', 'average_directional_movement')
    def average_directional_movement_index(self):  # TODO: implement
        pass

    # TODO: STOCHRSI

    # TODO: WILLR

    @Alias('cci', 'CCI', 'commodity_channel')
    def commodity_channel_index(self):  # TODO: implement
        pass

    # TODO: ADXR

    # TODO: APO

    # TODO: PPO

    # TODO: MOM

    # TODO: BOP

    @Alias('Aroon', 'AROON')
    def aroon(self):  # TODO: implement
        pass

    # TODO: CMO

    # TODO: ROC

    # TODO: ROCR

    @Alias('bbands', 'BBANDS', 'Bollinger_bands')
    def bollinger_bands(self):  # TODO: implement
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

    # TODO: MIDPOINT

    # TODO: MIDPRICE

    # TODO: SAR

    # TODO: TRANGE

    # TODO: ATR

    # TODO: NATR

    @Alias('obv', 'OBV', 'balance_volume')
    def on_balance_volume(self):  # TODO: implement
        pass

    # TODO: ADOSC

    # TODO: HT_TRENDLINE

    # TODO: HT_SINE

    # TODO: HT_TRENDMODE

    # TODO: HT_DCPERIOD

    # TODO: HT_DCPHASE

    # TODO: HT_PHASOR

    def _get_info(self):
        print(self.__gen_info)


if __name__ == '__main__':
    s = Stock('MSFT', period='1mo', interval='1wk')
    print(s.financials)

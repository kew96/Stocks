from django.db import models
from django.utils import timezone
from polymorphic.models import PolymorphicModel

from decimal import *
from datetime import date
from dateutil.relativedelta import relativedelta

from functions.errors.errors import NoTickerError
from functions.stock import ticker_search
import functions.stock
from functions.helper_functions.option_valuation import black_scholes_merton, binomial_pricing_tree

getcontext().prec = 2


# Create your models here.


class Portfolio(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, default='Test')
    cash = models.DecimalField(decimal_places=2, max_digits=100)
    inception = models.DateField(default=timezone.now)
    fee_rate = models.FloatField(default=0)

    def __str__(self):
        return f'{self.name} ${self.total_value()}'

    def total_value(self):
        open_trades = Trade.objects.filter(closed__isnull=True)
        unrealized_gains = sum([trade.current_value() for trade in open_trades])
        return self.cash + unrealized_gains

    # TODO: period performance

    # TODO: overall performance


# TODO: PortfolioValueHistory

class Trade(PolymorphicModel):
    # https://medium.com/@bhrigu/django-how-to-add-foreignkey-to-multiple-models-394596f06e84

    buy = 'Buy'
    sell = 'Sell'

    choices = (
        (buy, 'Buy'),
        (sell, 'Sell')
    )

    open = 'Open'
    close = 'Close'

    subchoices = (
        (open, 'Open'),
        (close, 'Close')
    )

    id = models.AutoField(primary_key=True)
    portfolio = models.ForeignKey('Portfolio', on_delete=models.CASCADE, related_name='trades')
    stock = models.ForeignKey('Stock', on_delete=models.CASCADE, related_name='trades')
    type = models.CharField(max_length=4, default=buy, choices=choices)
    subtype = models.CharField(max_length=5, default=open, choices=subchoices)
    initial_price = models.DecimalField(decimal_places=2, max_digits=100, default=0)
    shares = models.IntegerField(default=0)
    fee_rate = models.IntegerField(default=0)
    fee_cost = models.DecimalField(decimal_places=2, max_digits=100, default=0)
    trade_cost = models.DecimalField(decimal_places=2, max_digits=100, default=0)
    total_cost = models.DecimalField(decimal_places=2, max_digits=100, default=0)
    initiated = models.DateTimeField(default=timezone.now)
    closed = models.DateTimeField(blank=True, null=True)
    gain_loss_value = models.DecimalField(decimal_places=2, max_digits=100, blank=True, null=True)
    gain_loss_percent = models.DecimalField(decimal_places=6, max_digits=100, blank=True, null=True)
    reason = models.TextField(default='NA')


class TradeValueHistory(models.Model):
    id = models.AutoField(primary_key=True)
    trade = models.ForeignKey('Trade', on_delete=models.CASCADE, related_name='value_history')
    date = models.DateField(default=timezone.now)
    asset_price = models.DecimalField(decimal_places=2, max_digits=100)
    unrealized_gain_loss = models.DecimalField(decimal_places=2, max_digits=100, default=0)

    def __str__(self):
        return f'{self.trade}' + ' {' + f'{self.date}' + '}'


class Long(Trade):

    def __str__(self):
        return f'{self.stock.ticker}-Long-{self.type}({self.initiated})'

    def value(self, dt=date.today(), column='adj_close'):
        if dt == date.today():
            price = self.stock.get_bid()
        else:
            price = self.stock.get_price(dt=dt, column=column)
        return price - self.initial_price


class Short(Trade):

    def __str__(self):
        return f'{self.stock.ticker}-Short-{self.type}({self.initiated})'

    def value(self, dt=date.today(), column='adj_close'):
        if dt == date.today():
            price = self.stock.get_ask()
        else:
            price = self.stock.get_price(dt=dt, column=column)
        return self.initial_price - price


class Option(Trade):
    american = 'American'
    european = 'European'

    option_choices = (
        (american, 'American'),
        (european, 'European')
    )

    expiration = models.DateField(default=timezone.now() + relativedelta(months=3))
    strike = models.DecimalField(decimal_places=2, max_digits=100)
    option_cost = models.DecimalField(decimal_places=2, max_digits=100)
    option_type = models.CharField(max_length=8, default=american, choices=option_choices)


class LongPut(Option):

    def __str__(self):
        return f'{self.stock.ticker}-LongPut-{self.type}({self.expiration})-{self.strike}'

    def value(self, rf, column='Close', strike=None, steps=100, end_date=None, start_date=date.today(), tree=False):
        if strike is None:
            strike = self.strike
        if end_date is None:
            end_date = self.expiration
        dummy = functions.stock.HistoricalStock(ticker=self.stock.ticker,
                                                name=self.stock.name,
                                                start=start_date - relativedelta(years=2),
                                                end=start_date)
        if self.option_type == 'American':
            return binomial_pricing_tree(data=dummy.get_hist,
                                         column=column,
                                         strike=strike,
                                         rf=rf,
                                         steps=steps,
                                         end_date=end_date,
                                         start_date=start_date,
                                         tree=tree)
        else:
            return black_scholes_merton(data=dummy.get_hist,
                                        column=column,
                                        strike=strike,
                                        rf=rf,
                                        end_date=end_date,
                                        start_date=start_date,
                                        dividend_yield=self.stock.dividend_yield,
                                        call=False)


class LongCall(Option):

    def __str__(self):
        return f'{self.stock.ticker}-LongCall-{self.type}({self.expiration})-{self.strike}'

    def value(self, rf, column='Close', strike=None, end_date=None, start_date=date.today()):
        if strike is None:
            strike = self.strike
        if end_date is None:
            end_date = self.expiration
        dummy = functions.stock.HistoricalStock(ticker=self.stock.ticker,
                                                name=self.stock.name,
                                                start=start_date - relativedelta(years=2),
                                                end=start_date)
        return black_scholes_merton(data=dummy.get_hist,
                                    column=column,
                                    strike=strike,
                                    rf=rf,
                                    end_date=end_date,
                                    start_date=start_date,
                                    dividend_yield=self.stock.dividend_yield,
                                    call=True)


class ShortPut(Option):

    def __str__(self):
        return f'{self.stock.ticker}-ShortPut-{self.type}({self.expiration})-{self.strike}'

    def current_value(self, rf, column='Close', strike=None, steps=100, end_date=None, start_date=date.today(),
                      tree=False):
        if strike is None:
            strike = self.strike
        if end_date is None:
            end_date = self.expiration
        dummy = functions.stock.HistoricalStock(ticker=self.stock.ticker,
                                                name=self.stock.name,
                                                start=start_date - relativedelta(years=2),
                                                end=start_date)
        if self.option_type == 'American':
            if tree:
                stock_tree, option_tree = binomial_pricing_tree(data=dummy.get_hist,
                                                                column=column,
                                                                strike=strike,
                                                                rf=rf,
                                                                steps=steps,
                                                                end_date=end_date,
                                                                start_date=start_date,
                                                                tree=False)
                return stock_tree, -1 * option_tree
            else:
                return -binomial_pricing_tree(data=dummy.get_hist,
                                              column=column,
                                              strike=strike,
                                              rf=rf,
                                              steps=steps,
                                              end_date=end_date,
                                              start_date=start_date,
                                              tree=False)
        else:
            return -black_scholes_merton(data=dummy.get_hist,
                                         column=column,
                                         strike=strike,
                                         rf=rf,
                                         end_date=end_date,
                                         start_date=start_date,
                                         dividend_yield=self.stock.dividend_yield,
                                         call=False)


class ShortCall(Option):

    def __str__(self):
        return f'{self.stock.ticker}-ShortCall-{self.type}({self.expiration})-{self.strike}'

    def current_value(self, rf, column='Close', strike=None, end_date=None, start_date=date.today()):
        if strike is None:
            strike = self.strike
        if end_date is None:
            end_date = self.expiration
        dummy = functions.stock.HistoricalStock(ticker=self.stock.ticker,
                                                name=self.stock.name,
                                                start=start_date - relativedelta(years=2),
                                                end=start_date)
        return -black_scholes_merton(data=dummy.get_hist,
                                     column=column,
                                     strike=strike,
                                     rf=rf,
                                     end_date=end_date,
                                     start_date=start_date,
                                     dividend_yield=self.stock.dividend_yield,
                                     call=True)


class Stock(models.Model):
    ticker = models.CharField(max_length=10, unique=True, primary_key=True)
    name = models.CharField(max_length=200, unique=True)
    summary = models.TextField()
    sector = models.CharField(max_length=100)
    industry = models.CharField(max_length=300)
    dividend_rate = models.DecimalField(decimal_places=2, max_digits=100)
    beta = models.DecimalField(decimal_places=6, max_digits=100)
    trailing_PE = models.DecimalField(decimal_places=6, max_digits=100)
    market_cap = models.PositiveBigIntegerField(null=True)
    price_to_sales_12m = models.DecimalField(decimal_places=6, max_digits=100)
    forward_PE = models.DecimalField(decimal_places=6, max_digits=100)
    tradeable = models.BooleanField(default=True)
    dividend_yield = models.DecimalField(decimal_places=6, max_digits=100)
    forward_EPS = models.DecimalField(decimal_places=2, max_digits=100)
    profit_margin = models.DecimalField(decimal_places=10, max_digits=100)
    trailing_EPS = models.DecimalField(decimal_places=6, max_digits=100)

    def __str__(self):
        stock_obj = functions.stock.Stock(self.ticker, self.name, verbose=False)
        return f'{self.ticker} {stock_obj.bid}/{stock_obj.ask}'

    def save(self, *args, **kwargs):
        if self.ticker == '' and self.name == '':
            raise NoTickerError()
        elif self.ticker == '':
            self.ticker = ticker_search(self.name)[0][0]['1. symbol']
        elif self.name == '':
            self.name = ticker_search(self.ticker)[0][0]['2. name']

        stock_obj = functions.stock.Stock(self.ticker)

        if not self.summary:
            self.summary = stock_obj.summary
        if not self.sector:
            self.sector = stock_obj.sector
        if not self.industry:
            self.industry = stock_obj.industry
        if not self.dividend_rate:
            self.dividend_rate = stock_obj.dividend_rate
        if not self.beta:
            self.beta = stock_obj.beta
        if not self.trailing_PE:
            self.trailing_PE = stock_obj.trailing_PE
        if not self.market_cap:
            self.market_cap = stock_obj.market_cap
        if not self.price_to_sales_12m:
            self.price_to_sales_12m = stock_obj.price_to_sales_12m
        if not self.forward_PE:
            self.forward_PE = stock_obj.forward_PE
        if not self.tradeable:
            self.tradeable = stock_obj.tradeable
        if not self.dividend_yield:
            self.dividend_yield = stock_obj.dividend_yield
        if not self.forward_EPS:
            self.forward_EPS = stock_obj.forward_EPS
        if not self.profit_margin:
            self.profit_margin = stock_obj.profit_margin
        if not self.trailing_EPS:
            self.trailing_EPS = stock_obj.trailing_EPS

        super().save(*args, **kwargs)

    def get_price(self, dt, column='adj_close'):
        stock_data = functions.stock.HistoricalStock(ticker=self.ticker,
                                                     name=self.name,
                                                     start=dt,
                                                     end=dt + relativedelta(days=1),
                                                     verbose=False)
        price = Decimal(str(stock_data.get_hist.loc[:, column][0]))
        return price

    def get_bid(self):
        stock_data = functions.stock.Stock(ticker=self.ticker,
                                           name=self.name,
                                           verbose=False)
        price = Decimal(str(stock_data.bid))
        return price

    def get_ask(self):
        stock_data = functions.stock.Stock(ticker=self.ticker,
                                           name=self.name,
                                           verbose=False)
        price = Decimal(str(stock_data.ask))
        return price

from django.db import models
from django.utils import timezone
from polymorphic.models import PolymorphicModel

from decimal import *
from dateutil.relativedelta import relativedelta

from functions.errors.errors import NoTickerError
from functions.stock import ticker_search
import functions.stock


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


class Long(Trade):

    def __str__(self):
        return f'{self.stock.ticker}-Long-{self.type}({self.initiated})'

    def current_value(self):
        dummy = functions.stock.Stock(self.stock.ticker)
        current_price = Decimal(dummy.bid)
        return current_price - Decimal(str(self.initial_price))


class Short(Trade):

    def __str__(self):
        return f'{self.stock.ticker}-Short-{self.type}({self.initiated})'

    def current_value(self):
        dummy = functions.stock.Stock(self.stock.ticker)
        current_price = Decimal(dummy.ask)
        return Decimal(str(self.initial_price)) - current_price


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
    type = models.CharField(max_length=8, default=american, choices=option_choices)


class LongPut(Option):

    def __str__(self):
        return f'{self.stock.ticker}-LongPut-{self.type}({self.expiration})-{self.strike}'

    def current_value(self):  # TODO: Create American option pricing tree
        dummy = Stock(self.stock.ticker)
        current_price = Decimal(dummy.current['05. price'])
        option_value = self.strike - current_price - self.option_cost
        return max(option_value, self.option_cost)


class LongCall(Option):

    def __str__(self):
        return f'{self.stock.ticker}-LongCall-{self.type}({self.expiration})-{self.strike}'

    def current_value(self):  # TODO: Create American option pricing tree
        dummy = Stock(self.stock.ticker)
        current_price = Decimal(dummy.current['05. price'])
        option_value = current_price - self.strike - self.option_cost
        return max(option_value, self.option_cost)


class ShortPut(Option):

    def __str__(self):
        return f'{self.stock.ticker}-ShortPut-{self.type}({self.expiration})-{self.strike}'

    def current_value(self):  # TODO: Create American option pricing tree
        dummy = Stock(self.stock.ticker)
        current_price = Decimal(dummy.current['05. price'])
        option_value = current_price - self.strike + self.option_cost
        return min(option_value, self.option_cost)


class ShortCall(Option):

    def __str__(self):
        return f'{self.stock.ticker}-ShortCall-{self.type}({self.expiration})-{self.strike}'

    def current_value(self):  # TODO: Create American option pricing tree
        temp_stock = Stock(self.stock.ticker)
        current_price = Decimal(temp_stock.current['05. price'])
        option_value = self.strike - current_price + self.option_cost
        return min(option_value, self.option_cost)


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
    tradable = models.BooleanField(default=True)
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
        if not self.tradable:
            self.tradable = stock_obj.tradable
        if not self.dividend_yield:
            self.dividend_yield = stock_obj.dividend_yield
        if not self.forward_EPS:
            self.forward_EPS = stock_obj.forward_EPS
        if not self.profit_margin:
            self.profit_margin = stock_obj.profit_margin
        if not self.trailing_EPS:
            self.trailing_EPS = stock_obj.trailing_EPS

        super().save(*args, **kwargs)

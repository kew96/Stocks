from django.db import models
from django.contrib.contenttypes.models import ContentType
from polymorphic.models import PolymorphicModel

from datetime import datetime, date
from dateutil.relativedelta import relativedelta


# Create your models here.


class Portfolio(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, default='Test')
    cash = models.DecimalField(decimal_places=2, max_digits=100)
    inception = models.DateField(default=date.today())

    def __str__(self):
        return f'{self.name} ${self.total_value()}'

    def total_value(self):  # TODO: total value
        return self.cash

    # TODO: period performance

    # TODO: overall performance


class Trade(PolymorphicModel):  # TODO: finish
    # https://medium.com/@bhrigu/django-how-to-add-foreignkey-to-multiple-models-394596f06e84

    buy = 'Buy'
    sell = 'Sell'

    choices = (
        (buy, 'Buy'),
        (sell, 'Sell')
    )

    id = models.AutoField(primary_key=True)
    stock = models.ForeignKey('Stock', on_delete=models.CASCADE, related_name='trades')
    type = models.CharField(max_length=4, default=buy, choices=choices)
    fee_rate = models.IntegerField(default=0)
    fee_cost = models.DecimalField(decimal_places=2, max_digits=100)
    trade_cost = models.DecimalField(decimal_places=2, max_digits=100)
    total_cost = models.DecimalField(decimal_places=2, max_digits=100)
    initiated = models.DateTimeField(default=datetime.now())
    closed = models.DateTimeField(blank=True)
    reason = models.TextField()


class Long(Trade):  # TODO: finish

    def __str__(self):
        return f'{self.stock.ticker}-Long-{self.type}({self.initiated})'


class Short(Trade):  # TODO: finish

    def __str__(self):
        return f'{self.stock.ticker}-Short-{self.type}({self.initiated})'


class Put(Trade):  # TODO: finish
    expiration = models.DateField(default=date.today() + relativedelta(months=3))
    strike = models.DecimalField(decimal_places=2, max_digits=100)

    def __str__(self):
        return f'{self.stock.ticker}-Put-{self.type}({self.expiration})-{self.strike}'


class Call(Trade):  # TODO: finish
    expiration = models.DateField(default=date.today() + relativedelta(months=3))
    strike = models.DecimalField(decimal_places=2, max_digits=100)

    def __str__(self):
        return f'{self.stock.ticker}-Call-{self.type}({self.expiration})-{self.strike}'


class Stock(models.Model):  # TODO: finish
    ticker = models.CharField(max_length=10, unique=True, primary_key=True)
    name = models.CharField(max_length=200, unique=True)

    def __str__(self):
        return self.ticker

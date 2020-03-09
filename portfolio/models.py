from django.db import models
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey
from polymorphic.models import PolymorphicModel

from datetime import date


# Create your models here.


class Portfolio(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, default='Test')
    cash = models.FloatField()
    inception = models.DateField(default=date.today())

    def __str__(self):
        return self.name

    # TODO: total value

    # TODO: period performance

    # TODO: overall performance


class StockActivity(models.Model):
    long = 'Long'
    short = 'Short'
    put = 'Put'
    call = 'Call'

    choices = (
        (long, 'Long'),
        (short, 'Short'),
        (put, 'Put'),
        (call, 'Call')
    )

    buy = 'Buy'
    sell = 'Sell'

    subchoices = (
        (buy, 'Buy'),
        (sell, 'Sell')
    )

    id = models.AutoField(primary_key=True)
    stock = models.ForeignKey('Stock', on_delete=models.CASCADE, related_name='activity')
    trade = models.ForeignKey('Trade', on_delete=models.CASCADE)
    type = models.CharField(default=long, choices=choices)
    subtype = models.CharField(default=buy, choices=subchoices)


class Trade(PolymorphicModel):
    # https://medium.com/@bhrigu/django-how-to-add-foreignkey-to-multiple-models-394596f06e84
    id = models.AutoField(primary_key=True)
    # TODO: finish
    pass


class Long(Trade):
    # TODO: finish
    pass


class Short(Trade):
    # TODO: finish
    pass


class Put(Trade):
    # TODO: finish
    pass


class Call(Trade):
    # TODO: finish
    pass


class Stock(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField()
    trades = models.ManyToManyField(Trade, related_name='stock')
    # TODO: finish
    pass

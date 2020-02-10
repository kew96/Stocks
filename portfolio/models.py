from django.db import models
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey

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


class Stock(models.Model):
    # TODO: finish
    pass


class Trade(models.Model):
    # https://medium.com/@bhrigu/django-how-to-add-foreignkey-to-multiple-models-394596f06e84
    # TODO: finish
    pass

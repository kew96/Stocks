# Generated by Django 3.1.2 on 2020-10-05 19:57

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0025_long_longcall_longput_option_portfolio_short_shortcall_shortput_stock_trade'),
    ]

    operations = [
        migrations.AlterField(
            model_name='option',
            name='expiration',
            field=models.DateField(default=datetime.datetime(2021, 1, 5, 19, 57, 43, 371009, tzinfo=utc)),
        ),
    ]
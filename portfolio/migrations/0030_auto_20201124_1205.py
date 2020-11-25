# Generated by Django 3.1.2 on 2020-11-24 17:05

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0029_auto_20201123_2312'),
    ]

    operations = [
        migrations.AlterField(
            model_name='option',
            name='expiration',
            field=models.DateField(default=datetime.datetime(2021, 2, 24, 17, 5, 44, 767526, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='stock',
            name='market_cap',
            field=models.PositiveBigIntegerField(null=True),
        ),
    ]
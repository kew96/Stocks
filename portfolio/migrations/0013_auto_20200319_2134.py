# Generated by Django 3.0.3 on 2020-03-20 01:34

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0012_auto_20200319_2132'),
    ]

    operations = [
        migrations.AlterField(
            model_name='option',
            name='expiration',
            field=models.DateField(default=datetime.datetime(2020, 6, 20, 1, 34, 43, 937141, tzinfo=utc)),
        ),
    ]
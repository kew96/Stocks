# Generated by Django 3.1.2 on 2020-11-25 14:57

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0031_auto_20201125_0955'),
    ]

    operations = [
        migrations.AlterField(
            model_name='option',
            name='expiration',
            field=models.DateField(default=datetime.datetime(2021, 2, 25, 14, 57, 38, 453019, tzinfo=utc)),
        ),
    ]

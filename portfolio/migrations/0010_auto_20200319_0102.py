# Generated by Django 3.0.3 on 2020-03-19 05:02

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0009_auto_20200319_0100'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trade',
            name='initiated',
            field=models.DateTimeField(default=datetime.datetime(2020, 3, 19, 1, 2, 3, 968581)),
        ),
    ]
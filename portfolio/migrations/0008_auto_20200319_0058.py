# Generated by Django 3.0.3 on 2020-03-19 04:58

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0007_auto_20200319_0055'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trade',
            name='initiated',
            field=models.DateTimeField(default=datetime.datetime(2020, 3, 19, 0, 58, 0, 648589)),
        ),
    ]

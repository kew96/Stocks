# Generated by Django 3.0.3 on 2020-03-23 02:08

import datetime
from django.db import migrations, models
from django.utils.timezone import utc
import django.utils.timezone


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0016_auto_20200322_2206'),
    ]

    operations = [
        migrations.AlterField(
            model_name='option',
            name='expiration',
            field=models.DateField(default=datetime.datetime(2020, 6, 23, 2, 8, 35, 224415, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='portfolio',
            name='inception',
            field=models.DateField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='trade',
            name='initiated',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]

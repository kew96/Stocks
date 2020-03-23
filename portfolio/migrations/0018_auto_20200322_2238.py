# Generated by Django 3.0.3 on 2020-03-23 02:38

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0017_auto_20200322_2208'),
    ]

    operations = [
        migrations.AlterField(
            model_name='option',
            name='expiration',
            field=models.DateField(default=datetime.datetime(2020, 6, 23, 2, 38, 28, 663769, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='stock',
            name='industry',
            field=models.CharField(max_length=300),
        ),
    ]

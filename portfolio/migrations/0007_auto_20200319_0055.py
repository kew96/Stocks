# Generated by Django 3.0.3 on 2020-03-19 04:55

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0006_auto_20200319_0053'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trade',
            name='initiated',
            field=models.DateTimeField(default=datetime.datetime(2020, 3, 19, 0, 55, 1, 95182)),
        ),
        migrations.AlterField(
            model_name='trade',
            name='stock',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='trades',
                                    to='portfolio.Stock'),
        ),
    ]

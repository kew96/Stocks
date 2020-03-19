# Generated by Django 3.0.3 on 2020-03-19 04:34

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0002_auto_20200318_1923'),
    ]

    operations = [
        migrations.CreateModel(
            name='Option',
            fields=[
                ('trade_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Trade')),
                ('expiration', models.DateField(default=datetime.date(2020, 6, 19))),
                ('strike', models.DecimalField(decimal_places=2, max_digits=100)),
                ('option_cost', models.DecimalField(decimal_places=2, max_digits=100)),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.trade',),
        ),
        migrations.RemoveField(
            model_name='put',
            name='trade_ptr',
        ),
        migrations.AddField(
            model_name='trade',
            name='fee_cost',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=100),
        ),
        migrations.AddField(
            model_name='trade',
            name='fee_rate',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='trade',
            name='gain_loss',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=100, null=True),
        ),
        migrations.AddField(
            model_name='trade',
            name='initial_price',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=100),
        ),
        migrations.AddField(
            model_name='trade',
            name='shares',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='trade',
            name='total_cost',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=100),
        ),
        migrations.AddField(
            model_name='trade',
            name='trade_cost',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=100),
        ),
        migrations.AlterField(
            model_name='portfolio',
            name='inception',
            field=models.DateField(default=datetime.date(2020, 3, 19)),
        ),
        migrations.AlterField(
            model_name='trade',
            name='closed',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='trade',
            name='initiated',
            field=models.DateTimeField(default=datetime.datetime(2020, 3, 19, 0, 34, 12, 693026)),
        ),
        migrations.AlterField(
            model_name='trade',
            name='reason',
            field=models.TextField(default='NA'),
        ),
        migrations.CreateModel(
            name='LongCall',
            fields=[
                ('option_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Option')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.option',),
        ),
        migrations.CreateModel(
            name='LongPut',
            fields=[
                ('option_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Option')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.option',),
        ),
        migrations.CreateModel(
            name='ShortCall',
            fields=[
                ('option_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Option')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.option',),
        ),
        migrations.CreateModel(
            name='ShortPut',
            fields=[
                ('option_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Option')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.option',),
        ),
        migrations.DeleteModel(
            name='Call',
        ),
        migrations.DeleteModel(
            name='Put',
        ),
    ]

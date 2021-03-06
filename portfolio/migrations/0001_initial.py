# Generated by Django 3.0.3 on 2020-03-18 23:12

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='Portfolio',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(default='Test', max_length=100)),
                ('cash', models.DecimalField(decimal_places=2, max_digits=100)),
                ('inception', models.DateField(default=datetime.date(2020, 3, 18))),
            ],
        ),
        migrations.CreateModel(
            name='Stock',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=200, unique=True)),
                ('ticker', models.CharField(max_length=10, unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='Trade',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('type', models.CharField(choices=[('Buy', 'Buy'), ('Sell', 'Sell')], default='Buy', max_length=4)),
                ('initiated', models.DateTimeField(default=datetime.datetime(2020, 3, 18, 19, 12, 43, 731716))),
                ('closed', models.DateTimeField(blank=True)),
                ('reason', models.TextField()),
                ('polymorphic_ctype',
                 models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE,
                                   related_name='polymorphic_portfolio.trade_set+', to='contenttypes.ContentType')),
                ('stock', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='trades',
                                            to='portfolio.Stock')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
        ),
        migrations.CreateModel(
            name='Call',
            fields=[
                ('trade_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Trade')),
                ('expiration', models.DateField(default=datetime.date(2020, 6, 18))),
                ('strike', models.DecimalField(decimal_places=2, max_digits=100)),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.trade',),
        ),
        migrations.CreateModel(
            name='Long',
            fields=[
                ('trade_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Trade')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.trade',),
        ),
        migrations.CreateModel(
            name='Put',
            fields=[
                ('trade_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Trade')),
                ('expiration', models.DateField(default=datetime.date(2020, 6, 18))),
                ('strike', models.DecimalField(decimal_places=2, max_digits=100)),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.trade',),
        ),
        migrations.CreateModel(
            name='Short',
            fields=[
                ('trade_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='portfolio.Trade')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('portfolio.trade',),
        ),
    ]

# Generated by Django 3.0.3 on 2020-03-20 01:32

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0011_auto_20200319_2129'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trade',
            name='initiated',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]

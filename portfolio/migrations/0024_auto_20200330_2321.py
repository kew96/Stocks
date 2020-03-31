# Generated by Django 3.0.4 on 2020-03-31 03:21

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('portfolio', '0023_merge_20200330_2321'),
        ]

    operations = [
        migrations.RemoveField(
                 model_name='longcall',
                 name='option_ptr',
                 ),
        migrations.RemoveField(
                 model_name='longput',
                 name='option_ptr',
                 ),
        migrations.RemoveField(
                 model_name='option',
                 name='trade_ptr',
                 ),
        migrations.RemoveField(
                 model_name='short',
                 name='trade_ptr',
                 ),
        migrations.RemoveField(
                 model_name='shortcall',
                 name='option_ptr',
                 ),
        migrations.RemoveField(
                 model_name='shortput',
                 name='option_ptr',
                 ),
        migrations.RemoveField(
                 model_name='trade',
                 name='polymorphic_ctype',
                 ),
        migrations.RemoveField(
                 model_name='trade',
                 name='portfolio',
                 ),
        migrations.RemoveField(
                 model_name='trade',
                 name='stock',
                 ),
        migrations.DeleteModel(
                 name='Long',
                 ),
        migrations.DeleteModel(
                 name='LongCall',
                 ),
        migrations.DeleteModel(
                 name='LongPut',
                 ),
        migrations.DeleteModel(
                 name='Option',
                 ),
        migrations.DeleteModel(
                 name='Portfolio',
                 ),
        migrations.DeleteModel(
                 name='Short',
                 ),
        migrations.DeleteModel(
                 name='ShortCall',
                 ),
        migrations.DeleteModel(
                 name='ShortPut',
                 ),
        migrations.DeleteModel(
                 name='Stock',
                 ),
        migrations.DeleteModel(
                 name='Trade',
                 ),
        ]

# Generated by Django 4.0 on 2022-04-18 16:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gestionContact', '0007_remove_file_kine_file_bool_predict'),
    ]

    operations = [
        migrations.AlterField(
            model_name='predict',
            name='prediction',
            field=models.CharField(default='', max_length=50),
        ),
    ]

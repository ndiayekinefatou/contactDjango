# Generated by Django 4.0 on 2022-03-04 23:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('gestionContact', '0002_rename_file_file_filename'),
    ]

    operations = [
        migrations.AddField(
            model_name='file',
            name='prediction',
            field=models.CharField(default='', editable=False, max_length=50),
        ),
        migrations.AddField(
            model_name='file',
            name='transcription',
            field=models.CharField(default='', editable=False, max_length=200),
        ),
    ]
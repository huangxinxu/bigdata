# Generated by Django 3.2.23 on 2023-12-17 14:19

from django.db import migrations, models
import maeApp.models


class Migration(migrations.Migration):

    dependencies = [
        ('maeApp', '0002_auto_20231217_1245'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelpth',
            name='file',
            field=models.FileField(null=True, upload_to=maeApp.models.pth_directory_path),
        ),
    ]
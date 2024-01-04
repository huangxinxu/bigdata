from django.db import models
import os
import uuid


# Create your models here.
# YourApp/models.py

class Transaction(models.Model):
    date = models.DateField()
    volume = models.FloatField(null=True)
    average_price = models.FloatField(null=True)
    status = models.IntegerField()
    city = models.CharField(max_length=255, verbose_name='城市')

    def __str__(self):
        return f"{self.date} - Volume: {self.volume}, Avg. Price: {self.average_price}, Status: {self.status}"


def user_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    return os.path.join("data", filename)


class Xlsxdata(models.Model):
    file = models.FileField(upload_to=user_directory_path, null=True)

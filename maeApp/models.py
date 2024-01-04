#file_upload/models.py
from django.db import models
import os
import uuid

# Create your models here.
# Define user directory path
def user_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    return os.path.join("files", filename)

class Img(models.Model):
    file = models.FileField(upload_to=user_directory_path, null=True)

def pth_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format(uuid.uuid4().hex[:10], ext)
    return os.path.join("files/pth", filename)
class ModelPth(models.Model):
    file = models.FileField(upload_to=pth_directory_path, null=True)
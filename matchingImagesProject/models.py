from django.db import models

class Image(models.Model):
    name = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images')
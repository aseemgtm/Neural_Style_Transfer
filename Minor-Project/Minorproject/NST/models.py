from django.db import models

# Create your models here.
class User(models.Model):
    pic=models.ImageField(upload_to ="style");
class User1(models.Model):
    pic1=models.ImageField(upload_to ="content");

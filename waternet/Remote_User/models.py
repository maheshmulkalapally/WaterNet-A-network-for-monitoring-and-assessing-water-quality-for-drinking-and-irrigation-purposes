from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address = models.CharField(max_length=3000,null=True, blank=True)
    gender = models.CharField(max_length=300,null=True, blank=True)

class assessing_water_quality(models.Model):

    RID= models.CharField(max_length=3000)
    State= models.CharField(max_length=3000)
    District_Name= models.CharField(max_length=3000)
    Place_Name= models.CharField(max_length=3000)
    ph= models.CharField(max_length=3000)
    Hardness= models.CharField(max_length=3000)
    Solids= models.CharField(max_length=3000)
    Chloramines= models.CharField(max_length=3000)
    Sulfate= models.CharField(max_length=3000)
    Conductivity= models.CharField(max_length=3000)
    Organic_carbon= models.CharField(max_length=3000)
    Trihalomethanes= models.CharField(max_length=3000)
    Turbidity= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)




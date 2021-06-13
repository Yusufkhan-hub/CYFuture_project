from django.db import models

# Create your models here.


class tasks(models.Model):
    short_description = models.CharField(max_length=200)
    category = models.CharField(max_length=50)
    date_added = models.DateField(auto_now=True)
    due_date = models.DateField()
    is_completed = models.BooleanField()

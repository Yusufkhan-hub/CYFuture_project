from django.forms import fields
from .models import tasks
from django import forms



class TasksForm(forms.ModelForm):
    class Meta:
        model = tasks
        fields = "__all__"

        
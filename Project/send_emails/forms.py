from django import forms
from django.core.mail import message
from django.forms.widgets import ClearableFileInput



class SendEmail(forms.Form):
    email = forms.EmailField(required=True)
    subject = forms.CharField(max_length=120, required=True)
    attach = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple':True}))
    message = forms.CharField(widget=forms.Textarea)

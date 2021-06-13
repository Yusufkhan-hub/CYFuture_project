from django.urls import path
from .views import EmailAttchementView

urlpatterns = [
    path('send-email', EmailAttchementView.as_view(), name='emailattachment')
]
from django.urls import path
from . import views
urlpatterns = [
    path('ml_model_prediction',views.ML_Home,name="ml_home"),
    path('result_model',views.predicted_result,name="ml_result"),
]

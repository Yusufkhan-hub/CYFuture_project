from django.urls import path
from . import views

# app_name='to_do_app'
urlpatterns = [
    path('get-task',views.get_task,name="get-tasks"),
    path('post-task',views.post_task,name="post-tasks"),
    path('update-task/<int:id>/',views.update_task,name="update-tasks"),
    path('delete-task/<int:id>/',views.delete_task,name="delete-tasks"),
]

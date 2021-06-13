from django.shortcuts import redirect, render
from .forms import TasksForm
from .models import  tasks
from django.contrib import messages
# Create your views here.


def get_task(request):
    obj= tasks.objects.all()
    return render(request,'todo_app/to-do-get.html',{'obj':obj})

def post_task(request):
    if request.method=='POST':
        form = TasksForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request,"Task added successfully")
            return redirect('/')
        else:
            messages.error(request,"Something went wrong!! try again..!!")
            return redirect('/')
    else:
        form=TasksForm()

    return render(request,'todo_app/to-do-post.html',{'form':form})


def update_task(request,id):
    obj=tasks.objects.get(id=id)
    form = TasksForm(instance=obj)
    if request.method=="POST":
        form = TasksForm(request.POST,instance=obj)
        if form.is_valid():
            form.save()
            messages.success(request,'Task updated successfully..!!') 
            return redirect('/')
        else:
            messages.error(request,'Something went wrong, please try again..!!!')
            return redirect('/')
    else:
        form = TasksForm()
        return render(request,'todo_app/to-do-update.html',{'form':form})


def delete_task(request,id=id):
    obj = tasks.objects.get(id=id)
    if request.method=='POST':
        obj.delete()
        return redirect('/')
    return render(request,'todo_app/to-do-delete.html',{'obj':obj})


from django.contrib.auth import authenticate, login
from django.shortcuts import redirect, render
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import RegisterForm
#Create your views here.


def index(request):
    return render(request,'home.html')


def signup(request):
    
    if request.method=='POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username=form.cleaned_data.get('username')
            password=form.cleaned_data.get('password1')
            user=authenticate(username=username,password=password)
            login(request,user)
            messages.success(request,"User added succesfully")
            return redirect('index')
    else:
        form=RegisterForm()
    return render(request,'registration/signup.html',{'form':form})
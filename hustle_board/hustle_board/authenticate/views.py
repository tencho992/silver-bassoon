from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.views.decorators.clickjacking import xframe_options_exempt

def home(request):
    return render(request, 'authenticate/home.html', {})

# @xframe_options_exempt
# def test(request):
#     return render(request, 'authenticate/may.py', {})


def profile(request):
    print(request.user)
    return render(request, 'authenticate/profile.html', { "user" : request.user })

def login_user(request):
    if request.method == 'POST':
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, 'Under Development :)')
            return redirect('profile')
        else:
            messages.success(request, 'Error Logging in, Please try again...')
            return redirect('home')
    else:
        return render(request, 'authenticate/login.html', {})

def logout_user(request):
    logout(request)
    return redirect('home')

def register_user(request):
        if request.method == 'POST':
            form = UserCreationForm(request.POST)
            if form.is_valid():
                form.save()
                username = form.cleaned_data['username']
                password = form.cleaned_data['password1']
                user = authenticate(username=username, password=password)
                login(request, user)
                messages.success(request, ('You Have Registered'))
                return redirect('profile')
        else: 
            form = UserCreationForm
        context = { 'form' : form }
        return render(request, 'authenticate/register.html', context)
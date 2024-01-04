from django.http import HttpResponse

from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.forms.models import model_to_dict
from django.db.models import Q
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User, Group

def loginpage(request):
    ctx = {}
    return render(request, 'login.html', ctx)

def logincheck(request):
    ctx = {}
    username = request.GET['username']
    pwd = request.GET['password']
    user = authenticate(username=username, password=pwd)
    if user is not None:
        login(request, user)
        # identity = 0
        # if user.groups.filter(name='auditor').exists():
        #     identity = 1
        return render(request,"welcome.html",{'user':user})
    else:
        ctx = {'查无此人'}
        return render(request,"login.html",{'ctx':ctx})


def loginout(request):
    ctx = {}
    logout(request)
    return render(request, "login.html", {'ctx': ctx})

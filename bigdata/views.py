from django.http import HttpResponse

from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.forms.models import model_to_dict
from django.db.models import Q
from datetime import datetime
from django.contrib.auth.decorators import login_required

@login_required
def welcome(request):
    ctx = {}
    return render(request, 'welcome.html', ctx)


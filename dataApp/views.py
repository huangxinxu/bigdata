from django.http import HttpResponse

from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.forms.models import model_to_dict
from django.db.models import Q
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from .forms import TransactionForm, XlsxUploadModelForm, SearchTime
from .models import Transaction
from django.db.models import Q
import pandas as pd

@login_required
def showdata(request):
    ctx = {}
    if request.method == "POST":
        singledataform = TransactionForm(request.POST, request.FILES)
        xlsxform = XlsxUploadModelForm(request.POST, request.FILES)
        if singledataform.is_valid():
            singledata = singledataform.save(commit=False)
            singledata.status = 0
            singledata.save()
        if xlsxform.is_valid():
            xlsxfile = xlsxform.save()
            #读取xlsx然后存
            pdata = pd.read_excel(io=xlsxfile.file)
            for row in pdata.itertuples():
                t = Transaction(
                    date = row[2],
                    volume= row[4],
                    average_price = row[3],
                    status = 0,
                    city = row[1],
                )
                t.save()
    else:
        singledataform = TransactionForm()
        xlsxform = XlsxUploadModelForm()

    return render(request, 'showdata.html', {"TransactionForm": singledataform, "XlsxUploadModelForm": xlsxform})

@login_required
def backend(request):
    ctx = {}
    if request.method == "POST":
        stime = SearchTime(request.POST)
    else:
        stime = SearchTime()
    return render(request, 'backend.html', {"SearchTime":stime})

def TPdata(request):
    nrow = int(request.GET['rows'])
    page = int(request.GET['page'])
    s = (page - 1) * nrow
    e = page * nrow + 1

    start_datetime = request.GET['start_datetime']
    end_datetime = request.GET['end_datetime']

    q = Q(id__gt = 0)
    q = q & Q(status=0)
    if start_datetime != "" and end_datetime != "":
        start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_datetime, '%Y-%m-%d')
        q4 = Q(date__range=[start_datetime,end_datetime])
        q = q & q4
    else:
        if start_datetime != "":
            start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d')
            q4 = Q(date__gte=start_datetime)
            q = q & q4
        if end_datetime != "":
            end_datetime = datetime.strptime(end_datetime, '%Y-%m-%d')
            q4 = Q(date__lte=end_datetime)
            q = q & q4

    temp = Transaction.objects.filter(q)

    result = {}

    result["total"] = len(temp)

    result["rows"] = []
    num = 0
    for t in temp:
        num = num + 1
        if num > (page - 1) * nrow and num < page * nrow + 1:
            a = model_to_dict(t)
            a['status']='待修改'
            result['rows'].append(a)

    return JsonResponse(result)
@csrf_exempt
def UpdateT(request):
    ctx = {}
    if request.method == "POST":
        selected_ids = request.POST.getlist('selected_ids[]', [])
        for i in selected_ids:
            t = Transaction.objects.filter(id=i)[0]
            t.status = 1
            t.save()
    stime = SearchTime()
    return render(request, 'backend.html', {"success":"success"})
@csrf_exempt
def DelT(request):
    ctx = {}
    if request.method == "POST":
        selected_ids = request.POST.getlist('selected_ids[]', [])
        for i in selected_ids:
            t = Transaction.objects.filter(id=i)[0]
            t.delete()

    stime = SearchTime()
    return render(request, 'backend.html', {"success": "success"})
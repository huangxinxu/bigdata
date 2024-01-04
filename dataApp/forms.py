# YourApp/forms.py
from django import forms
from .models import Transaction, Xlsxdata
from django.forms import widgets

class DateInput(forms.DateInput):
    input_type = 'date'

class TransactionForm(forms.ModelForm):
    date = forms.DateField(widget=DateInput)
    class Meta:
        model = Transaction
        fields = ['date', 'city', 'volume', 'average_price',]

        labels = {
            'date': '交易日期',
            'volume': '交易量',
            'city': '城市',
            'average_price': '交易均价',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['volume'].required = False
        self.fields['average_price'].required = False


class SearchTime(forms.Form):
    cdate = forms.DateField(widget=widgets.TextInput(attrs={'type':'date'}), label='开始时间')
    mdate = forms.DateField(widget=widgets.TextInput(attrs={'type':'date'}), label='结束时间')


class XlsxUploadModelForm(forms.ModelForm):
    class Meta:
        model = Xlsxdata
        fields = ('file',)
        widgets = {
            'file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }

    def clean_file(self):
        file = self.cleaned_data['file']
        ext = file.name.split('.')[-1].lower()
        if ext not in ["xls","xlsx"]:
            raise forms.ValidationError("Only xls and xlsx files are allowed.")
        # return cleaned data is very important.
        return file
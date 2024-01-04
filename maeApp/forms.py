#file_upload/forms.py

from django import forms
from .models import Img, ModelPth

# Regular form
# class FileUploadForm(forms.Form):
#     file = forms.FileField(widget=forms.ClearableFileInput(attrs={'class': 'form-control'}))
#     upload_method = forms.CharField(label="Upload Method", max_length=20,
#                                    widget=forms.TextInput(attrs={'class': 'form-control'}))
#     def clean_file(self):
#         file = self.cleaned_data['file']
#         ext = file.name.split('.')[-1].lower()
#         if ext not in ["jpg", "png", "tif"]:
#             raise forms.ValidationError("Only jpg, png and tif files are allowed.")
#         # return cleaned data is very important.
#         return file

# Model form
class ImgUploadModelForm(forms.ModelForm):
    class Meta:
        model = Img
        fields = ('file',)
        widgets = {
            'file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }

    def clean_file(self):
        file = self.cleaned_data['file']
        ext = file.name.split('.')[-1].lower()
        if ext not in ["jpg", "png", "tif"]:
            raise forms.ValidationError("Only jpg, png and tif files are allowed.")
        # return cleaned data is very important.
        return file

class ModelPthUploadModelForm(forms.ModelForm):
    class Meta:
        model = ModelPth
        fields = ('file',)
        widgets = {
            'file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }

    def clean_file(self):
        file = self.cleaned_data['file']
        ext = file.name.split('.')[-1].lower()
        if ext not in ["pth"]:
            raise forms.ValidationError("Only pth files are allowed.")
        # return cleaned data is very important.
        return file
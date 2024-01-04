"""bigdata URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from django.conf.urls import url

from . import views
from dataApp import views as data_views
from loginApp import views as login_views
from django.views.generic import RedirectView
from maeApp import views as mae_views
from ndviApp import views as ndvi_views
from django.views.static import serve
from django.conf import settings


urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$', login_views.loginpage),
    url(r'^welcome', ndvi_views.ndvi),
    # url(r'^ndvi', ndvi_views.ndvi),
    url(r'^logincheck', login_views.logincheck),
    url(r'^loginout', login_views.loginout),
    url(r'^login', login_views.loginpage),
    url(r'^showdata', data_views.showdata),
    url(r'^backend', data_views.backend),
    url(r'^mae', mae_views.mae,name='mae'),
    url(r'^TPdata', data_views.TPdata),
    url(r'^UpdateT', data_views.UpdateT),
    url(r'^DelT', data_views.DelT),

    url(r'media/(?P<path>.*)',serve,{'document_root':settings.MEDIA_ROOT}),
    path('<path:undefined_path>', RedirectView.as_view(url='/', permanent=False)),

]
"""Minorproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from NST import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('input.html', views.input , name='input'),
    path('input1.html', views.input1 , name='input1'),
    path('upload',views.uploadImage,name='uploadImage'),
    path('upload1',views.uploadImage1,name='uploadImage1'),
    path('input2.html', views.input2 , name='input2'),
    path('output.html', views.output , name='output'),
]  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

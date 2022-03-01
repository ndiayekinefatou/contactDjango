from django.urls import path
#from django.conf.urls import url
from . import views
from . views import Home

urlpatterns = [
    path('', views.Home.as_view(), name='upload')

        ]

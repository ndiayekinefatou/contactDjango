from django.urls import path
from . import views
from . views import Home

urlpatterns = [
    path('', views.Home.as_view(), name='upload'),
    #path('get',views.HomeGet.as_view())


        ]

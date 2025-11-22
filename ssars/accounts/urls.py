from django.urls import path
from . import views
urlpatterns=[path('signupaccount/',views.signupaccount,name='signupaccount'),path('loginaccount/',views.loginaccount,name='loginaccount'),path('logoutaccount/',views.logoutaccount,name='logoutaccount')
             ]
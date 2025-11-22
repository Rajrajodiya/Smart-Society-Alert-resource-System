from django.urls import path
from . import views

urlpatterns = [
    path('', views.weather_home, name='weather'),
    path('predict/', views.weather_predict, name='weather_predict'),
    path('visualize/', views.weather_visualize, name='weather_visualize'),
    path('analysis/', views.weather_analysis, name='weather_analysis'),
    path('alerts/', views.weather_alerts, name='weather_alerts'),
    path('api/data/', views.weather_api_data, name='weather_api_data'),
] 
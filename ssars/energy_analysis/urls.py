from django.urls import path
from . import views

app_name = 'energy_analysis'

urlpatterns = [
    path('', views.energy_dashboard, name='dashboard'),
    path('visualization/', views.energy_visualization, name='visualization'),
    path('train-model/', views.train_models, name='train_model'),
    path('predict/', views.predict_energy_consumption, name='predict'),
    path('alerts/', views.energy_alerts, name='alerts'),
    path('data/', views.energy_data_list, name='data_list'),
    path('utility-monitor/', views.utility_monitor, name='utility_monitor'),
    path('api/stats/', views.api_energy_stats, name='api_stats'),
] 
from django.shortcuts import render
from news.models import News
from weather.models import WeatherData
from weather.utils import WeatherAPI

def home(request):
    """Main home page view with current weather data"""
    # Get latest news and weather data
    latest_news = News.objects.all().order_by('-date')[:3]
    latest_weather = WeatherData.objects.all().order_by('-dt_txt')[:5]
    
    # Get current weather for dashboard
    weather_api = WeatherAPI()
    current_weather = weather_api.get_current_weather('Mumbai')  # Default city
    
    context = {
        'latest_news': latest_news,
        'latest_weather': latest_weather,
        'current_weather': current_weather,
        'news_count': News.objects.count(),
        'weather_count': WeatherData.objects.count(),
    }
    return render(request, 'home.html', context) 
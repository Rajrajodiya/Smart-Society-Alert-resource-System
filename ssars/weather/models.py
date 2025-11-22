from django.db import models
from django.utils import timezone

# Create your models here.
class WeatherData(models.Model):
    dt_txt = models.DateTimeField()
    city = models.CharField(max_length=100)
    temp = models.FloatField()
    temp_min = models.FloatField(default=0.0)
    temp_max = models.FloatField(default=0.0)
    humidity = models.FloatField()
    pressure = models.IntegerField()
    speed = models.FloatField()
    visibility = models.FloatField()
    weather_main = models.CharField(max_length=50, default='')
    weather_description = models.CharField(max_length=100, default='')
    country = models.CharField(max_length=50, default='')
    feels_like = models.FloatField(default=0.0)
    wind_deg = models.FloatField(default=0.0)
    
    # Enhanced fields for analysis
    weather_type = models.CharField(max_length=50, default='', blank=True)
    weather_subtype = models.CharField(max_length=100, default='', blank=True)
    uv_index = models.FloatField(default=0.0)
    cloudiness = models.IntegerField(default=0)
    precipitation = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.city} - {self.dt_txt} - {self.weather_main}"
    
    class Meta:
        ordering = ['-dt_txt']
        indexes = [
            models.Index(fields=['city', 'dt_txt']),
            models.Index(fields=['weather_type']),
            models.Index(fields=['temp']),
        ]

class WeatherAlert(models.Model):
    ALERT_TYPES = [
        ('storm', 'Storm Warning'),
        ('rain', 'Heavy Rain'),
        ('heat', 'Heat Wave'),
        ('cold', 'Cold Wave'),
        ('fog', 'Fog Alert'),
        ('wind', 'High Wind'),
        ('flood', 'Flood Warning'),
        ('snow', 'Snow Alert'),
        ('thunderstorm', 'Thunderstorm'),
        ('dust', 'Dust Storm'),
        ('smog', 'Smog Alert'),
    ]
    
    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField()
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS)
    city = models.CharField(max_length=100)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Additional fields for better alert management
    affected_area = models.CharField(max_length=200, blank=True)
    recommendations = models.TextField(blank=True)
    source = models.CharField(max_length=100, default='System Generated')
    
    def __str__(self):
        return f"{self.title} - {self.city}"
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['city', 'is_active']),
            models.Index(fields=['alert_type', 'severity']),
        ]

class WeatherAnalysis(models.Model):
    """Store weather analysis results"""
    city = models.CharField(max_length=100)
    analysis_date = models.DateField()
    avg_temp = models.FloatField()
    max_temp = models.FloatField()
    min_temp = models.FloatField()
    avg_humidity = models.FloatField()
    total_precipitation = models.FloatField(default=0.0)
    dominant_weather_type = models.CharField(max_length=50, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.city} - {self.analysis_date}"
    
    class Meta:
        ordering = ['-analysis_date']
        unique_together = ['city', 'analysis_date']
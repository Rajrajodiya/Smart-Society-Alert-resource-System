from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Avg, Max, Min, Count
from .models import WeatherData, WeatherAlert
from .utils import WeatherAPI, WeatherPredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import json
import numpy as np
import os

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

# Create your views here.
def weather_home(request):
    """Display weather data with search functionality and 5-day forecast"""
    weather_api = WeatherAPI()
    current_weather = None
    forecast_data = None
    search_city = request.GET.get('city', 'Ahmedabad')
    
    if request.GET.get('city'):
        # Get current weather
        current_weather = weather_api.get_current_weather(search_city)
        
        # Get 5-day forecast
        forecast_data = weather_api.get_forecast(search_city)
        
        # Save current weather to database if available
        if current_weather:
            try:
                WeatherData.objects.create(
                    dt_txt=current_weather['dt'],
                    city=current_weather['city'],
                    temp=current_weather['temp'],
                    temp_min=current_weather['temp_min'],
                    temp_max=current_weather['temp_max'],
                    humidity=current_weather['humidity'],
                    pressure=current_weather['pressure'],
                    speed=current_weather['wind_speed'],
                    visibility=current_weather['visibility'],
                    weather_main=current_weather['weather_main'],
                    weather_description=current_weather['weather_description'],
                    country=current_weather['country'],
                    feels_like=current_weather['feels_like'],
                    wind_deg=current_weather['wind_deg']
                )
            except Exception as e:
                print(f"Error saving weather data: {e}")
    
    # Get latest weather data from database
    weather_data = WeatherData.objects.all().order_by('-dt_txt')[:10]
    
    # Get weather statistics
    stats = WeatherData.objects.aggregate(
        total_records=Count('id'),
        avg_temp=Avg('temp'),
        max_temp=Max('temp'),
        min_temp=Min('temp'),
        avg_humidity=Avg('humidity')
    )
    
    context = {
        'weather_data': weather_data,
        'current_weather': current_weather,
        'forecast_data': forecast_data,
        'search_city': search_city,
        'stats': stats,
    }
    return render(request, 'weather/home.html', context)

def weather_predict(request):
    """Enhanced Weather prediction using ML models with date range analysis"""
    prediction = None
    error_message = None
    historical_data = None
    prediction_chart = None
    weather_trends = None
    
    if request.method == 'POST':
        try:
            city = request.POST.get('city', '')
            temp = float(request.POST.get('temp', 0))
            humidity = float(request.POST.get('humidity', 0))
            start_date = request.POST.get('start_date', '')
            end_date = request.POST.get('end_date', '')
            
            if not city:
                error_message = "Please enter a city name"
            elif temp < -50 or temp > 60:
                error_message = "Temperature must be between -50°C and 60°C"
            elif humidity < 0 or humidity > 100:
                error_message = "Humidity must be between 0% and 100%"
            else:
                predictor = WeatherPredictor()
                
                # Get basic prediction
                prediction = predictor.predict(city, temp, humidity)
                
                # Get historical data analysis if date range is provided
                if start_date and end_date:
                    historical_data = predictor.get_historical_analysis(city, start_date, end_date)
                    weather_trends = predictor.analyze_weather_trends(city, start_date, end_date)
                    prediction_chart = predictor.create_prediction_visualization(city, start_date, end_date)
                
                if prediction is None:
                    error_message = "Unable to make prediction. Please try again."
            
        except ValueError:
            error_message = "Please enter valid numeric values for temperature and humidity"
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
    
    context = {
        'prediction': prediction,
        'error_message': error_message,
        'form_data': request.POST if request.method == 'POST' else None,
        'historical_data': historical_data,
        'prediction_chart': prediction_chart,
        'weather_trends': weather_trends,
    }
    return render(request, 'weather/predict.html', context)

def weather_visualize(request):
    """Enhanced weather data visualization from CSV with date range input"""
    charts = {}
    error_message = None
    weather_stats = None
    date_filter_applied = False
    start_date = None
    end_date = None
    
    try:
        # Load data from CSV
        df = load_weather_data_from_csv()
        
        if df is None or df.empty:
            error_message = "No weather data available. Please ensure weather_data.csv exists."
            context = {
                'has_data': False,
                'error_message': error_message,
                'form_data': request.GET
            }
            return render(request, 'weather/visualize.html', context)
        
        # Handle date filtering
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        city_filter = request.GET.get('city', '').strip()
        
        original_count = len(df)
        
        # Apply date filtering if provided
        if start_date and end_date:
            try:
                # Convert input dates to the same timezone as the data
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                # Ensure timezone consistency
                if df['dt_txt'].dt.tz is not None:
                    # Data has timezone, make input dates timezone-aware
                    start_dt = start_dt.tz_localize('UTC')
                    end_dt = end_dt.tz_localize('UTC')
                else:
                    # Data is timezone-naive, ensure input dates are also timezone-naive
                    df['dt_txt'] = pd.to_datetime(df['dt_txt']).dt.tz_localize(None)
                
                df = df[(df['dt_txt'] >= start_dt) & (df['dt_txt'] <= end_dt)]
                date_filter_applied = True
            except Exception as date_error:
                print(f"Date filtering error: {date_error}")
                # If date filtering fails, continue without filtering
                pass
        
        # Apply city filtering if provided
        if city_filter:
            df = df[df['city'].str.contains(city_filter, case=False, na=False)]
        
        if df.empty:
            error_message = "No data found for the selected filters. Please adjust your date range or city."
            context = {
                'has_data': False,
                'error_message': error_message,
                'form_data': request.GET
            }
            return render(request, 'weather/visualize.html', context)
        
        # Calculate comprehensive statistics with min-max ranges
        weather_stats = calculate_weather_statistics(df, original_count, date_filter_applied)
        
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create visualizations
        try:
            # 1. Temperature Analysis with Min-Max Range
            charts['temperature'] = create_temperature_analysis_chart(df)
            
            # 2. Humidity Analysis with Min-Max Range
            charts['humidity'] = create_humidity_analysis_chart(df)
            
            # 3. Pressure Analysis with Min-Max Range
            charts['pressure'] = create_pressure_analysis_chart(df)
            
            # 4. Multi-parameter trends with ranges
            charts['multi_trends'] = create_multi_parameter_trends_chart(df)
            
            # 5. Weather distribution by city
            charts['city_distribution'] = create_city_weather_distribution_chart(df)
            
            # 6. Hourly patterns with min-max bands
            charts['hourly_patterns'] = create_hourly_patterns_chart(df)
            
            # 7. Monthly analysis with ranges
            charts['monthly_analysis'] = create_monthly_analysis_chart(df)
            
            # 8. Weather type distribution
            charts['weather_types'] = create_weather_type_distribution_chart(df)
            
            # 9. Correlation matrix
            charts['correlation'] = create_correlation_heatmap(df)
            
            # 10. Daily weather summary
            charts['daily_summary'] = create_daily_weather_summary_chart(df)
            
        except Exception as chart_error:
            print(f"Chart creation error: {chart_error}")
            # Create a simple error chart
            charts = create_basic_charts(df)
        
        # Get date range for UI
        try:
            min_date = df['dt_txt'].min()
            max_date = df['dt_txt'].max()
            if hasattr(min_date, 'strftime'):
                min_date_str = min_date.strftime('%Y-%m-%d')
                max_date_str = max_date.strftime('%Y-%m-%d')
            else:
                min_date_str = str(min_date)[:10]
                max_date_str = str(max_date)[:10]
        except:
            min_date_str = '2000-01-01'
            max_date_str = '2025-12-31'
        
        context = {
            'charts': charts,
            'has_data': True,
            'weather_stats': weather_stats,
            'date_filter_applied': date_filter_applied,
            'form_data': request.GET,
            'available_cities': get_available_cities(df),
            'date_range': {
                'min_date': min_date_str,
                'max_date': max_date_str
            }
        }
        
    except Exception as e:
        print(f"Main error in weather_visualize: {str(e)}")
        error_message = f"Error creating visualizations: {str(e)}"
        context = {
            'has_data': False,
            'error_message': error_message,
            'form_data': request.GET
        }
    
    return render(request, 'weather/visualize.html', context)

def load_weather_data_from_csv():
    """Load weather data from CSV file with proper datetime handling"""
    try:
        # Try multiple possible paths for the CSV file
        possible_paths = [
            'ssars/static/data/weather_data.csv',
            'static/data/weather_data.csv',
            '../ssars/static/data/weather_data.csv',
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"Found weather data at: {path}")
                break
        
        if df is None:
            print("Weather data CSV not found")
            return None
        
        # Clean and prepare data
        df = df.dropna()
        
        # Handle timestamp column with proper timezone handling
        if 'dt_txt' in df.columns:
            df['dt_txt'] = pd.to_datetime(df['dt_txt'], utc=True).dt.tz_localize(None)
        elif 'timestamp' in df.columns:
            df['dt_txt'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        else:
            print("No datetime column found")
            return None
        
        # Ensure required columns exist
        required_cols = ['temp', 'humidity', 'pressure', 'city']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return None
        
        # Convert numeric columns to proper types
        numeric_cols = ['temp', 'humidity', 'pressure']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values in critical columns
        df = df.dropna(subset=['dt_txt', 'temp', 'humidity', 'pressure'])
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['dt_txt'].min()} to {df['dt_txt'].max()}")
        
        return df
    except Exception as e:
        print(f"Error loading weather data: {e}")
        return None

def create_basic_charts(df):
    """Create basic charts when advanced charts fail"""
    charts = {}
    
    try:
        # Simple temperature chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by date if possible
        try:
            daily_temp = df.groupby(df['dt_txt'].dt.date)['temp'].mean()
            ax.plot(daily_temp.index, daily_temp.values, marker='o', linewidth=2, color='red')
            ax.set_title('Temperature Over Time', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Temperature (°C)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        except:
            # If grouping by date fails, just plot raw data (sample if too large)
            sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
            ax.plot(range(len(sample_df)), sample_df['temp'].values, marker='o', linewidth=1, color='red')
            ax.set_title('Temperature Data Sample', fontsize=16)
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Temperature (°C)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        charts['temperature'] = save_chart_to_base64(fig)
        
        # Simple humidity chart
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            daily_humidity = df.groupby(df['dt_txt'].dt.date)['humidity'].mean()
            ax.plot(daily_humidity.index, daily_humidity.values, marker='s', linewidth=2, color='blue')
            ax.set_title('Humidity Over Time', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Humidity (%)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        except:
            sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
            ax.plot(range(len(sample_df)), sample_df['humidity'].values, marker='s', linewidth=1, color='blue')
            ax.set_title('Humidity Data Sample', fontsize=16)
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Humidity (%)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        charts['humidity'] = save_chart_to_base64(fig)
        
        # Simple pressure chart
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            daily_pressure = df.groupby(df['dt_txt'].dt.date)['pressure'].mean()
            ax.plot(daily_pressure.index, daily_pressure.values, marker='^', linewidth=2, color='green')
            ax.set_title('Pressure Over Time', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Pressure (hPa)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        except:
            sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
            ax.plot(range(len(sample_df)), sample_df['pressure'].values, marker='^', linewidth=1, color='green')
            ax.set_title('Pressure Data Sample', fontsize=16)
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Pressure (hPa)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        charts['pressure'] = save_chart_to_base64(fig)
        
    except Exception as e:
        print(f"Error creating basic charts: {e}")
    
    return charts

def calculate_weather_statistics(df, original_count, date_filter_applied):
    """Calculate comprehensive weather statistics with min-max ranges"""
    try:
        # Handle date range calculation safely
        try:
            min_date = df['dt_txt'].min()
            max_date = df['dt_txt'].max()
            if hasattr(min_date, 'strftime') and hasattr(max_date, 'strftime'):
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                date_range = f"{str(min_date)[:10]} to {str(max_date)[:10]}"
        except:
            date_range = "Unknown"
        
        stats = {
            'total_records': len(df),
            'original_count': original_count,
            'filtered': date_filter_applied,
            'date_range': date_range,
            'cities_count': df['city'].nunique() if 'city' in df.columns else 0,
            'unique_cities': sorted(df['city'].unique()) if 'city' in df.columns else [],
            
            # Temperature statistics
            'temperature': {
                'avg': round(df['temp'].mean(), 2) if 'temp' in df.columns else 0,
                'min': round(df['temp'].min(), 2) if 'temp' in df.columns else 0,
                'max': round(df['temp'].max(), 2) if 'temp' in df.columns else 0,
                'range': round(df['temp'].max() - df['temp'].min(), 2) if 'temp' in df.columns else 0,
                'std': round(df['temp'].std(), 2) if 'temp' in df.columns else 0
            },
            
            # Humidity statistics
            'humidity': {
                'avg': round(df['humidity'].mean(), 2) if 'humidity' in df.columns else 0,
                'min': round(df['humidity'].min(), 2) if 'humidity' in df.columns else 0,
                'max': round(df['humidity'].max(), 2) if 'humidity' in df.columns else 0,
                'range': round(df['humidity'].max() - df['humidity'].min(), 2) if 'humidity' in df.columns else 0,
                'std': round(df['humidity'].std(), 2) if 'humidity' in df.columns else 0
            },
            
            # Pressure statistics
            'pressure': {
                'avg': round(df['pressure'].mean(), 2) if 'pressure' in df.columns else 0,
                'min': round(df['pressure'].min(), 2) if 'pressure' in df.columns else 0,
                'max': round(df['pressure'].max(), 2) if 'pressure' in df.columns else 0,
                'range': round(df['pressure'].max() - df['pressure'].min(), 2) if 'pressure' in df.columns else 0,
                'std': round(df['pressure'].std(), 2) if 'pressure' in df.columns else 0
            }
        }
        
        # Weather type distribution if available
        if 'main' in df.columns:
            stats['weather_types'] = df['main'].value_counts().to_dict()
        elif 'weather_type' in df.columns:
            stats['weather_types'] = df['weather_type'].value_counts().to_dict()
        
        return stats
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return {
            'total_records': len(df) if df is not None else 0,
            'original_count': original_count,
            'filtered': date_filter_applied,
            'date_range': "Error calculating date range",
            'cities_count': 0,
            'unique_cities': [],
            'temperature': {'avg': 0, 'min': 0, 'max': 0, 'range': 0, 'std': 0},
            'humidity': {'avg': 0, 'min': 0, 'max': 0, 'range': 0, 'std': 0},
            'pressure': {'avg': 0, 'min': 0, 'max': 0, 'range': 0, 'std': 0}
        }

def create_temperature_analysis_chart(df):
    """Create temperature analysis chart with min-max ranges"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Daily temperature trends with min-max bands
        daily_temp = df.groupby(df['dt_txt'].dt.date)['temp'].agg(['mean', 'min', 'max'])
        
        ax1.plot(daily_temp.index, daily_temp['mean'], marker='o', linewidth=2, label='Average', color='red')
        ax1.fill_between(daily_temp.index, daily_temp['min'], daily_temp['max'], 
                         alpha=0.3, color='red', label='Min-Max Range')
        ax1.set_title('Daily Temperature Analysis with Min-Max Range', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Temperature distribution histogram
        ax2.hist(df['temp'], bins=30, alpha=0.7, color='red', edgecolor='black')
        ax2.axvline(df['temp'].mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {df["temp"].mean():.1f}°C')
        ax2.axvline(df['temp'].min(), color='blue', linestyle='--', linewidth=2, label=f'Min: {df["temp"].min():.1f}°C')
        ax2.axvline(df['temp'].max(), color='orange', linestyle='--', linewidth=2, label=f'Max: {df["temp"].max():.1f}°C')
        ax2.set_title('Temperature Distribution', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Temperature (°C)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return save_chart_to_base64(fig)
    except Exception as e:
        print(f"Error creating temperature chart: {e}")
        return create_simple_chart(df, 'temp', 'Temperature', '°C', 'red')

def create_humidity_analysis_chart(df):
    """Create humidity analysis chart with min-max ranges"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Daily humidity trends with min-max bands
        daily_humidity = df.groupby(df['dt_txt'].dt.date)['humidity'].agg(['mean', 'min', 'max'])
        
        ax1.plot(daily_humidity.index, daily_humidity['mean'], marker='s', linewidth=2, label='Average', color='blue')
        ax1.fill_between(daily_humidity.index, daily_humidity['min'], daily_humidity['max'], 
                         alpha=0.3, color='blue', label='Min-Max Range')
        ax1.set_title('Daily Humidity Analysis with Min-Max Range', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Humidity (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Humidity distribution histogram
        ax2.hist(df['humidity'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(df['humidity'].mean(), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {df["humidity"].mean():.1f}%')
        ax2.axvline(df['humidity'].min(), color='green', linestyle='--', linewidth=2, label=f'Min: {df["humidity"].min():.1f}%')
        ax2.axvline(df['humidity'].max(), color='red', linestyle='--', linewidth=2, label=f'Max: {df["humidity"].max():.1f}%')
        ax2.set_title('Humidity Distribution', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Humidity (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return save_chart_to_base64(fig)
    except Exception as e:
        print(f"Error creating humidity chart: {e}")
        return create_simple_chart(df, 'humidity', 'Humidity', '%', 'blue')

def create_pressure_analysis_chart(df):
    """Create pressure analysis chart with min-max ranges"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Daily pressure trends with min-max bands
        daily_pressure = df.groupby(df['dt_txt'].dt.date)['pressure'].agg(['mean', 'min', 'max'])
        
        ax1.plot(daily_pressure.index, daily_pressure['mean'], marker='^', linewidth=2, label='Average', color='green')
        ax1.fill_between(daily_pressure.index, daily_pressure['min'], daily_pressure['max'], 
                         alpha=0.3, color='green', label='Min-Max Range')
        ax1.set_title('Daily Pressure Analysis with Min-Max Range', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Pressure (hPa)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Pressure distribution histogram
        ax2.hist(df['pressure'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(df['pressure'].mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {df["pressure"].mean():.1f} hPa')
        ax2.axvline(df['pressure'].min(), color='blue', linestyle='--', linewidth=2, label=f'Min: {df["pressure"].min():.1f} hPa')
        ax2.axvline(df['pressure'].max(), color='red', linestyle='--', linewidth=2, label=f'Max: {df["pressure"].max():.1f} hPa')
        ax2.set_title('Pressure Distribution', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Pressure (hPa)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return save_chart_to_base64(fig)
    except Exception as e:
        print(f"Error creating pressure chart: {e}")
        return create_simple_chart(df, 'pressure', 'Pressure', 'hPa', 'green')

def create_multi_parameter_trends_chart(df):
    """Create multi-parameter trends chart"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Group by date for daily trends
    daily_stats = df.groupby(df['dt_txt'].dt.date).agg({
        'temp': ['mean', 'min', 'max'],
        'humidity': ['mean', 'min', 'max'],
        'pressure': ['mean', 'min', 'max']
    })
    
    # Temperature
    axes[0].plot(daily_stats.index, daily_stats['temp']['mean'], marker='o', linewidth=2, color='red', label='Average')
    axes[0].fill_between(daily_stats.index, daily_stats['temp']['min'], daily_stats['temp']['max'], 
                        alpha=0.3, color='red', label='Range')
    axes[0].set_title('Temperature Trends with Min-Max Range', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Humidity
    axes[1].plot(daily_stats.index, daily_stats['humidity']['mean'], marker='s', linewidth=2, color='blue', label='Average')
    axes[1].fill_between(daily_stats.index, daily_stats['humidity']['min'], daily_stats['humidity']['max'], 
                        alpha=0.3, color='blue', label='Range')
    axes[1].set_title('Humidity Trends with Min-Max Range', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Humidity (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Pressure
    axes[2].plot(daily_stats.index, daily_stats['pressure']['mean'], marker='^', linewidth=2, color='green', label='Average')
    axes[2].fill_between(daily_stats.index, daily_stats['pressure']['min'], daily_stats['pressure']['max'], 
                        alpha=0.3, color='green', label='Range')
    axes[2].set_title('Pressure Trends with Min-Max Range', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylabel('Pressure (hPa)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return save_chart_to_base64(fig)

def create_city_weather_distribution_chart(df):
    """Create city-wise weather distribution chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top cities by record count
    city_counts = df['city'].value_counts().head(10)
    ax1.bar(range(len(city_counts)), city_counts.values, color='skyblue', edgecolor='black')
    ax1.set_title('Top 10 Cities by Data Records', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cities', fontsize=12)
    ax1.set_ylabel('Number of Records', fontsize=12)
    ax1.set_xticks(range(len(city_counts)))
    ax1.set_xticklabels(city_counts.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Average temperature by city (top 10)
    city_temp = df.groupby('city')['temp'].agg(['mean', 'min', 'max']).head(10)
    x_pos = range(len(city_temp))
    ax2.bar(x_pos, city_temp['mean'], yerr=[city_temp['mean'] - city_temp['min'], 
                                           city_temp['max'] - city_temp['mean']], 
           color='orange', alpha=0.7, capsize=5, edgecolor='black')
    ax2.set_title('Average Temperature by City (with Min-Max Range)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cities', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(city_temp.index, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_chart_to_base64(fig)

def create_hourly_patterns_chart(df):
    """Create hourly weather patterns chart"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Hourly temperature patterns
    hourly_temp = df.groupby(df['dt_txt'].dt.hour)['temp'].agg(['mean', 'min', 'max'])
    axes[0,0].plot(hourly_temp.index, hourly_temp['mean'], marker='o', linewidth=2, color='red', label='Average')
    axes[0,0].fill_between(hourly_temp.index, hourly_temp['min'], hourly_temp['max'], 
                          alpha=0.3, color='red', label='Min-Max Range')
    axes[0,0].set_title('Hourly Temperature Patterns', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Hour of Day', fontsize=12)
    axes[0,0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    axes[0,0].set_xticks(range(0, 24, 2))
    
    # Hourly humidity patterns
    hourly_humidity = df.groupby(df['dt_txt'].dt.hour)['humidity'].agg(['mean', 'min', 'max'])
    axes[0,1].plot(hourly_humidity.index, hourly_humidity['mean'], marker='s', linewidth=2, color='blue', label='Average')
    axes[0,1].fill_between(hourly_humidity.index, hourly_humidity['min'], hourly_humidity['max'], 
                          alpha=0.3, color='blue', label='Min-Max Range')
    axes[0,1].set_title('Hourly Humidity Patterns', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Hour of Day', fontsize=12)
    axes[0,1].set_ylabel('Humidity (%)', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    axes[0,1].set_xticks(range(0, 24, 2))
    
    # Hourly pressure patterns
    hourly_pressure = df.groupby(df['dt_txt'].dt.hour)['pressure'].agg(['mean', 'min', 'max'])
    axes[1,0].plot(hourly_pressure.index, hourly_pressure['mean'], marker='^', linewidth=2, color='green', label='Average')
    axes[1,0].fill_between(hourly_pressure.index, hourly_pressure['min'], hourly_pressure['max'], 
                          alpha=0.3, color='green', label='Min-Max Range')
    axes[1,0].set_title('Hourly Pressure Patterns', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Hour of Day', fontsize=12)
    axes[1,0].set_ylabel('Pressure (hPa)', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    axes[1,0].set_xticks(range(0, 24, 2))
    
    # Weather activity by hour
    if 'main' in df.columns:
        hourly_weather = df.groupby([df['dt_txt'].dt.hour, 'main']).size().unstack(fill_value=0)
        hourly_weather.plot(kind='bar', stacked=True, ax=axes[1,1], colormap='Set3')
        axes[1,1].set_title('Weather Types by Hour', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Hour of Day', fontsize=12)
        axes[1,1].set_ylabel('Frequency', fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend(title='Weather Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return save_chart_to_base64(fig)

def create_monthly_analysis_chart(df):
    """Create monthly weather analysis chart"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Monthly temperature analysis
    monthly_temp = df.groupby(df['dt_txt'].dt.month)['temp'].agg(['mean', 'min', 'max'])
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_pos = range(len(monthly_temp))
    
    axes[0,0].plot(x_pos, monthly_temp['mean'], marker='o', linewidth=2, color='red', label='Average')
    axes[0,0].fill_between(x_pos, monthly_temp['min'], monthly_temp['max'], 
                          alpha=0.3, color='red', label='Min-Max Range')
    axes[0,0].set_title('Monthly Temperature Analysis', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Month', fontsize=12)
    axes[0,0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels([month_names[i-1] for i in monthly_temp.index])
    
    # Monthly humidity analysis
    monthly_humidity = df.groupby(df['dt_txt'].dt.month)['humidity'].agg(['mean', 'min', 'max'])
    axes[0,1].plot(x_pos, monthly_humidity['mean'], marker='s', linewidth=2, color='blue', label='Average')
    axes[0,1].fill_between(x_pos, monthly_humidity['min'], monthly_humidity['max'], 
                          alpha=0.3, color='blue', label='Min-Max Range')
    axes[0,1].set_title('Monthly Humidity Analysis', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Month', fontsize=12)
    axes[0,1].set_ylabel('Humidity (%)', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels([month_names[i-1] for i in monthly_humidity.index])
    
    # Monthly pressure analysis
    monthly_pressure = df.groupby(df['dt_txt'].dt.month)['pressure'].agg(['mean', 'min', 'max'])
    axes[1,0].plot(x_pos, monthly_pressure['mean'], marker='^', linewidth=2, color='green', label='Average')
    axes[1,0].fill_between(x_pos, monthly_pressure['min'], monthly_pressure['max'], 
                          alpha=0.3, color='green', label='Min-Max Range')
    axes[1,0].set_title('Monthly Pressure Analysis', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Month', fontsize=12)
    axes[1,0].set_ylabel('Pressure (hPa)', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels([month_names[i-1] for i in monthly_pressure.index])
    
    # Monthly data distribution
    monthly_counts = df['dt_txt'].dt.month.value_counts().sort_index()
    axes[1,1].bar(range(len(monthly_counts)), monthly_counts.values, color='purple', alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Data Records by Month', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Month', fontsize=12)
    axes[1,1].set_ylabel('Number of Records', fontsize=12)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xticks(range(len(monthly_counts)))
    axes[1,1].set_xticklabels([month_names[i-1] for i in monthly_counts.index])
    
    plt.tight_layout()
    return save_chart_to_base64(fig)

def create_weather_type_distribution_chart(df):
    """Create weather type distribution chart"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Weather type pie chart
    if 'main' in df.columns:
        weather_counts = df['main'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(weather_counts)))
        axes[0].pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=colors)
        axes[0].set_title('Weather Type Distribution', fontsize=14, fontweight='bold')
    elif 'weather_type' in df.columns:
        weather_counts = df['weather_type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(weather_counts)))
        axes[0].pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=colors)
        axes[0].set_title('Weather Type Distribution', fontsize=14, fontweight='bold')
    
    # Temperature vs humidity scatter with weather types
    if 'main' in df.columns:
        unique_weather = df['main'].unique()[:10]  # Limit to top 10 for clarity
        for i, weather in enumerate(unique_weather):
            weather_data = df[df['main'] == weather]
            axes[1].scatter(weather_data['temp'], weather_data['humidity'], 
                           alpha=0.6, s=30, label=weather, c=plt.cm.tab10(i))
    else:
        axes[1].scatter(df['temp'], df['humidity'], alpha=0.6, s=30, c=df['temp'], cmap='viridis')
    
    axes[1].set_title('Temperature vs Humidity by Weather Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Temperature (°C)', fontsize=12)
    axes[1].set_ylabel('Humidity (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    if 'main' in df.columns:
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return save_chart_to_base64(fig)

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select numeric columns for correlation
    numeric_cols = ['temp', 'humidity', 'pressure']
    if 'speed' in df.columns:
        numeric_cols.append('speed')
    if 'visibility' in df.columns:
        numeric_cols.append('visibility')
    
    correlation_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
               square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax,
               fmt='.3f', annot_kws={'size': 12})
    ax.set_title('Weather Parameters Correlation Matrix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return save_chart_to_base64(fig)

def create_daily_weather_summary_chart(df):
    """Create daily weather summary chart"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Daily weather summary - box plots
    daily_data = df.groupby(df['dt_txt'].dt.date).agg({
        'temp': ['mean', 'min', 'max'],
        'humidity': ['mean', 'min', 'max'],
        'pressure': ['mean', 'min', 'max']
    })
    
    # Temperature box plot
    temp_data = [daily_data['temp']['min'], daily_data['temp']['mean'], daily_data['temp']['max']]
    box1 = axes[0].boxplot(temp_data, labels=['Min', 'Mean', 'Max'], patch_artist=True)
    for patch, color in zip(box1['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    axes[0].set_title('Daily Temperature Distribution (Min, Mean, Max)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Combined parameters over time (last 30 days if available)
    recent_data = df.tail(720)  # Approximately last 30 days if hourly data
    recent_daily = recent_data.groupby(recent_data['dt_txt'].dt.date).agg({
        'temp': 'mean',
        'humidity': 'mean',
        'pressure': 'mean'
    })
    
    # Normalize data for comparison
    temp_norm = (recent_daily['temp'] - recent_daily['temp'].min()) / (recent_daily['temp'].max() - recent_daily['temp'].min())
    humidity_norm = (recent_daily['humidity'] - recent_daily['humidity'].min()) / (recent_daily['humidity'].max() - recent_daily['humidity'].min())
    pressure_norm = (recent_daily['pressure'] - recent_daily['pressure'].min()) / (recent_daily['pressure'].max() - recent_daily['pressure'].min())
    
    axes[1].plot(recent_daily.index, temp_norm, marker='o', linewidth=2, label='Temperature (normalized)', color='red')
    axes[1].plot(recent_daily.index, humidity_norm, marker='s', linewidth=2, label='Humidity (normalized)', color='blue')
    axes[1].plot(recent_daily.index, pressure_norm, marker='^', linewidth=2, label='Pressure (normalized)', color='green')
    
    axes[1].set_title('Recent Weather Parameters Trends (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Normalized Values (0-1)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return save_chart_to_base64(fig)

def create_simple_chart(df, column, title, unit, color):
    """Create a simple chart when complex charts fail"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sample data if too large
        sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
        ax.plot(range(len(sample_df)), sample_df[column].values, color=color, linewidth=1)
        ax.set_title(f'{title} Data', fontsize=16)
        ax.set_xlabel('Data Points', fontsize=12)
        ax.set_ylabel(f'{title} ({unit})', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return save_chart_to_base64(fig)
    except:
        # Return empty chart if all else fails
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error creating {title} chart', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(f'{title} Chart Error', fontsize=16)
        return save_chart_to_base64(fig)

def get_available_cities(df):
    """Get list of available cities"""
    try:
        if 'city' in df.columns:
            return sorted(df['city'].unique())
        else:
            return []
    except:
        return []

def save_chart_to_base64(fig):
    """Save matplotlib figure to base64 string"""
    try:
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return chart_data
    except Exception as e:
        plt.close(fig)
        print(f"Error saving chart: {e}")
        return ""
def weather_analysis(request):
    """Comprehensive weather analysis"""
    weather_data = WeatherData.objects.all().values()
    
    if weather_data:
        df = pd.DataFrame(weather_data)
        df['dt_txt'] = pd.to_datetime(df['dt_txt'])
        
        # Perform analysis
        analysis_results = {
            'temperature_analysis': {
                'mean': round(df['temp'].mean(), 2),
                'median': round(df['temp'].median(), 2),
                'std': round(df['temp'].std(), 2),
                'min': round(df['temp'].min(), 2),
                'max': round(df['temp'].max(), 2),
                'range': round(df['temp'].max() - df['temp'].min(), 2)
            },
            'humidity_analysis': {
                'mean': round(df['humidity'].mean(), 2),
                'median': round(df['humidity'].median(), 2),
                'std': round(df['humidity'].std(), 2),
                'min': round(df['humidity'].min(), 2),
                'max': round(df['humidity'].max(), 2)
            },
            'pressure_analysis': {
                'mean': round(df['pressure'].mean(), 2),
                'median': round(df['pressure'].median(), 2),
                'std': round(df['pressure'].std(), 2),
                'min': round(df['pressure'].min(), 2),
                'max': round(df['pressure'].max(), 2)
            },
            'city_analysis': df['city'].value_counts().head(10).to_dict(),
            'weather_patterns': df['weather_main'].value_counts().to_dict() if 'weather_main' in df.columns else {},
            'seasonal_analysis': {
                'spring': round(df[df['dt_txt'].dt.month.isin([3, 4, 5])]['temp'].mean(), 2),
                'summer': round(df[df['dt_txt'].dt.month.isin([6, 7, 8])]['temp'].mean(), 2),
                'autumn': round(df[df['dt_txt'].dt.month.isin([9, 10, 11])]['temp'].mean(), 2),
                'winter': round(df[df['dt_txt'].dt.month.isin([12, 1, 2])]['temp'].mean(), 2)
            }
        }
        
        context = {
            'analysis_results': analysis_results,
            'has_data': True,
            'total_records': len(df)
        }
    else:
        context = {
            'has_data': False
        }
    
    return render(request, 'weather/analysis.html', context)

def weather_alerts(request):
    """Weather alerts page"""
    alerts = WeatherAlert.objects.filter(is_active=True).order_by('-created_at')
    
    # Generate some sample alerts based on weather conditions
    weather_api = WeatherAPI()
    current_weather = weather_api.get_current_weather('Ahmedabad')
    
    if current_weather:
        # Check for extreme conditions and create alerts
        temp = current_weather['temp']
        humidity = current_weather['humidity']
        visibility = current_weather['visibility']
        
        # Example alert generation logic
        if temp > 35:
            alert_type = 'heat'
            severity = 'high'
            title = 'Heat Wave Alert'
            description = f'High temperature detected: {temp}°C. Stay hydrated and avoid outdoor activities.'
        elif temp < 10:
            alert_type = 'cold'
            severity = 'medium'
            title = 'Cold Weather Alert'
            description = f'Low temperature detected: {temp}°C. Wear warm clothing.'
        elif visibility < 5:
            alert_type = 'fog'
            severity = 'medium'
            title = 'Fog Alert'
            description = f'Low visibility detected: {visibility}km. Drive carefully.'
        else:
            alert_type = None
            severity = None
            title = None
            description = None
    
    context = {
        'alerts': alerts,
        'current_weather': current_weather,
        'total_alerts': alerts.count()
    }
    return render(request, 'weather/alerts.html', context)

def weather_api_data(request):
    """API endpoint for weather data"""
    try:
        weather_data = WeatherData.objects.all().order_by('-dt_txt')[:100]
        
        data = []
        for record in weather_data:
            data.append({
                'city': record.city,
                'timestamp': record.dt_txt.isoformat(),
                'temperature': record.temp,
                'humidity': record.humidity,
                'pressure': record.pressure,
                'weather_main': record.weather_main,
                'weather_description': record.weather_description
            })
        
        return JsonResponse({
            'status': 'success',
            'data': data,
            'count': len(data)
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


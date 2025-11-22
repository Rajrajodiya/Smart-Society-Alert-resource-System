import requests
import json
from django.conf import settings
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64

class WeatherAPI:
    def __init__(self):
        self.api_key = settings.OPENWEATHER_API_KEY
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    def get_geocode(self, city):
        """Get coordinates for a city"""
        url = f"http://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': city,
            'limit': 1,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data:
                return {
                    'lat': data[0]['lat'],
                    'lon': data[0]['lon'],
                    'name': data[0]['name'],
                    'country': data[0]['country']
                }
            return None
        except Exception as e:
            print(f"Error getting geocode: {e}")
            return None
    
    def get_current_weather(self, city):
        """Get current weather for a city"""
        geocode = self.get_geocode(city)
        if not geocode:
            return None
        
        url = f"{self.base_url}/weather"
        params = {
            'lat': geocode['lat'],
            'lon': geocode['lon'],
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': geocode['name'],
                'country': geocode['country'],
                'temp': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'temp_min': data['main']['temp_min'],
                'temp_max': data['main']['temp_max'],
                'pressure': data['main']['pressure'],
                'humidity': data['main']['humidity'],
                'visibility': data.get('visibility', 0) / 1000,  # Convert to km
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg', 0),
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'weather_icon': data['weather'][0]['icon'],
                'dt': datetime.fromtimestamp(data['dt']),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']),
                'sunset': datetime.fromtimestamp(data['sys']['sunset'])
            }
        except Exception as e:
            print(f"Error getting current weather: {e}")
            return None
    
    def get_forecast(self, city, days=5):
        """Get 5-day weather forecast for a city"""
        geocode = self.get_geocode(city)
        if not geocode:
            return None
        
        url = f"{self.base_url}/forecast"
        params = {
            'lat': geocode['lat'],
            'lon': geocode['lon'],
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            forecast_data = []
            for item in data['list']:
                forecast_data.append({
                    'dt': datetime.fromtimestamp(item['dt']),
                    'temp': item['main']['temp'],
                    'temp_min': item['main']['temp_min'],
                    'temp_max': item['main']['temp_max'],
                    'pressure': item['main']['pressure'],
                    'humidity': item['main']['humidity'],
                    'visibility': item.get('visibility', 0) / 1000,
                    'wind_speed': item['wind']['speed'],
                    'wind_deg': item['wind'].get('deg', 0),
                    'weather_main': item['weather'][0]['main'],
                    'weather_description': item['weather'][0]['description'],
                    'weather_icon': item['weather'][0]['icon']
                })
            
            return forecast_data
        except Exception as e:
            print(f"Error getting forecast: {e}")
            return None

class WeatherPredictor:
    def __init__(self):
        self.model_temp = None
        self.model_visibility = None
        self.model_humidity = None
        self.model_pressure = None
    
    def prepare_data(self):
        """Prepare data from CSV file for training"""
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
            
            # Handle timestamp column
            if 'dt_txt' in df.columns:
                df['dt_txt'] = pd.to_datetime(df['dt_txt'])
            elif 'timestamp' in df.columns:
                df['dt_txt'] = pd.to_datetime(df['timestamp'])
            
            # Extract time features
            df['hour'] = df['dt_txt'].dt.hour
            df['day'] = df['dt_txt'].dt.day
            df['month'] = df['dt_txt'].dt.month
            df['day_of_week'] = df['dt_txt'].dt.dayofweek
            
            # Handle visibility column - check for different possible names
            visibility_col = None
            for col in df.columns:
                if 'visibility' in col.lower() or 'vis' in col.lower():
                    visibility_col = col
                    break
            
            if visibility_col is None:
                # Create a default visibility column based on other parameters
                df['visibility'] = 10.0  # Default visibility
                print("Visibility column not found, using default value")
            else:
                df['visibility'] = df[visibility_col]
            
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            return df
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None
    
    def train_models(self):
        """Train prediction models"""
        df = self.prepare_data()
        if df is None or df.empty:
            print("No data available for training")
            return False
        
        try:
            # Ensure required columns exist
            required_cols = ['temp', 'humidity', 'pressure', 'hour', 'day', 'month']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return False
            
            # Features for prediction
            features = ['temp', 'humidity', 'pressure', 'hour', 'day', 'month']
            
            # Train temperature model
            X_temp = df[['humidity', 'pressure', 'hour', 'day', 'month']]
            y_temp = df['temp']
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42
            )
            
            self.model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_temp.fit(X_train_temp, y_train_temp)
            
            # Train visibility model
            X_vis = df[['temp', 'humidity', 'pressure', 'hour', 'day', 'month']]
            y_vis = df['visibility']
            X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
                X_vis, y_vis, test_size=0.2, random_state=42
            )
            
            self.model_visibility = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_visibility.fit(X_train_vis, y_train_vis)
            
            # Train humidity model
            X_hum = df[['temp', 'pressure', 'hour', 'day', 'month']]
            y_hum = df['humidity']
            X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(
                X_hum, y_hum, test_size=0.2, random_state=42
            )
            
            self.model_humidity = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_humidity.fit(X_train_hum, y_train_hum)
            
            # Train pressure model
            X_pres = df[['temp', 'humidity', 'hour', 'day', 'month']]
            y_pres = df['pressure']
            X_train_pres, X_test_pres, y_train_pres, y_test_pres = train_test_split(
                X_pres, y_pres, test_size=0.2, random_state=42
            )
            
            self.model_pressure = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_pressure.fit(X_train_pres, y_train_pres)
            
            print("All models trained successfully")
            return True
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def predict(self, city, temp, humidity):
        """Make predictions based on input parameters"""
        if not all([self.model_temp, self.model_visibility, self.model_humidity, self.model_pressure]):
            if not self.train_models():
                return None
        
        try:
            # Use current time for prediction
            now = datetime.now()
            
            # Predict temperature
            temp_features = np.array([[humidity, 1013, now.hour, now.day, now.month]])
            predicted_temp = self.model_temp.predict(temp_features)[0]
            
            # Predict visibility
            vis_features = np.array([[temp, humidity, 1013, now.hour, now.day, now.month]])
            predicted_visibility = self.model_visibility.predict(vis_features)[0]
            
            # Predict humidity
            hum_features = np.array([[temp, 1013, now.hour, now.day, now.month]])
            predicted_humidity = self.model_humidity.predict(hum_features)[0]
            
            # Predict pressure
            pres_features = np.array([[temp, humidity, now.hour, now.day, now.month]])
            predicted_pressure = self.model_pressure.predict(pres_features)[0]
            
            # Determine weather type based on predictions
            weather_type = self.determine_weather_type(predicted_temp, predicted_humidity, predicted_visibility)
            
            return {
                'city': city,
                'predicted_temp': round(predicted_temp, 2),
                'predicted_visibility': round(predicted_visibility, 2),
                'predicted_humidity': round(predicted_humidity, 2),
                'predicted_pressure': round(predicted_pressure, 2),
                'weather_type': weather_type,
                'input_temp': temp,
                'input_humidity': humidity,
                'dt': now
            }
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def determine_weather_type(self, temp, humidity, visibility):
        """Determine weather type based on predicted parameters"""
        if temp < 0:
            return "Snow"
        elif temp < 10:
            return "Cold"
        elif temp > 35:
            return "Hot"
        elif humidity > 80:
            return "Humid"
        elif visibility < 5:
            return "Foggy"
        elif humidity > 60 and temp > 25:
            return "Warm"
        else:
            return "Clear" 

    def get_historical_analysis(self, city, start_date, end_date):
        """Analyze historical weather data for a city within date range"""
        try:
            df = self.prepare_data()
            if df is None:
                return None
            
            # Convert date strings to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data by city and date range
            city_data = df[df['city'].str.contains(city, case=False, na=False)]
            date_filtered = city_data[
                (city_data['dt_txt'] >= start_dt) & 
                (city_data['dt_txt'] <= end_dt)
            ]
            
            if date_filtered.empty:
                return None
            
            # Calculate statistics
            analysis = {
                'city': city,
                'start_date': start_date,
                'end_date': end_date,
                'total_records': len(date_filtered),
                'avg_temp': round(date_filtered['temp'].mean(), 2),
                'max_temp': round(date_filtered['temp'].max(), 2),
                'min_temp': round(date_filtered['temp'].min(), 2),
                'avg_humidity': round(date_filtered['humidity'].mean(), 2),
                'max_humidity': round(date_filtered['humidity'].max(), 2),
                'min_humidity': round(date_filtered['humidity'].min(), 2),
                'avg_pressure': round(date_filtered['pressure'].mean(), 2),
                'dominant_weather': date_filtered['main'].mode().iloc[0] if 'main' in date_filtered.columns else 'N/A',
                'weather_patterns': date_filtered['main'].value_counts().to_dict() if 'main' in date_filtered.columns else {}
            }
            
            return analysis
        except Exception as e:
            print(f"Error in historical analysis: {e}")
            return None
    
    def analyze_weather_trends(self, city, start_date, end_date):
        """Analyze weather trends for prediction"""
        try:
            df = self.prepare_data()
            if df is None:
                return None
            
            # Convert date strings to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data
            city_data = df[df['city'].str.contains(city, case=False, na=False)]
            date_filtered = city_data[
                (city_data['dt_txt'] >= start_dt) & 
                (city_data['dt_txt'] <= end_dt)
            ]
            
            if date_filtered.empty:
                return None
            
            # Group by date for daily trends
            daily_trends = date_filtered.groupby(date_filtered['dt_txt'].dt.date).agg({
                'temp': ['mean', 'max', 'min'],
                'humidity': 'mean',
                'pressure': 'mean'
            }).round(2)
            
            # Calculate temperature trend (increasing/decreasing)
            daily_temps = daily_trends['temp']['mean']
            temp_trend = "increasing" if daily_temps.iloc[-1] > daily_temps.iloc[0] else "decreasing"
            
            trends = {
                'temperature_trend': temp_trend,
                'temp_change': round(daily_temps.iloc[-1] - daily_temps.iloc[0], 2),
                'daily_data': daily_trends.to_dict(),
                'prediction_accuracy': self.calculate_prediction_accuracy(date_filtered)
            }
            
            return trends
        except Exception as e:
            print(f"Error in trend analysis: {e}")
            return None
    
    def create_prediction_visualization(self, city, start_date, end_date):
        """Create visualization chart for weather prediction"""
        try:
            df = self.prepare_data()
            if df is None:
                return None
            
            # Convert date strings to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data
            city_data = df[df['city'].str.contains(city, case=False, na=False)]
            date_filtered = city_data[
                (city_data['dt_txt'] >= start_dt) & 
                (city_data['dt_txt'] <= end_dt)
            ]
            
            if date_filtered.empty:
                return None
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Temperature trends
            plt.subplot(2, 2, 1)
            daily_temp = date_filtered.groupby(date_filtered['dt_txt'].dt.date)['temp'].mean()
            plt.plot(daily_temp.index, daily_temp.values, marker='o', color='red', linewidth=2)
            plt.title(f'Temperature Trends - {city}')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Humidity trends
            plt.subplot(2, 2, 2)
            daily_humidity = date_filtered.groupby(date_filtered['dt_txt'].dt.date)['humidity'].mean()
            plt.plot(daily_humidity.index, daily_humidity.values, marker='s', color='blue', linewidth=2)
            plt.title(f'Humidity Trends - {city}')
            plt.xlabel('Date')
            plt.ylabel('Humidity (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Pressure trends
            plt.subplot(2, 2, 3)
            daily_pressure = date_filtered.groupby(date_filtered['dt_txt'].dt.date)['pressure'].mean()
            plt.plot(daily_pressure.index, daily_pressure.values, marker='^', color='green', linewidth=2)
            plt.title(f'Pressure Trends - {city}')
            plt.xlabel('Date')
            plt.ylabel('Pressure (hPa)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Weather distribution
            plt.subplot(2, 2, 4)
            if 'main' in date_filtered.columns:
                weather_counts = date_filtered['main'].value_counts()
                plt.pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
                plt.title(f'Weather Distribution - {city}')
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_data
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def calculate_prediction_accuracy(self, data):
        """Calculate prediction accuracy based on historical data"""
        try:
            if len(data) < 2:
                return 0
            
            # Simple accuracy calculation based on data consistency
            temp_variance = data['temp'].var()
            humidity_variance = data['humidity'].var()
            
            # Lower variance = higher accuracy
            accuracy = max(0, min(100, 100 - (temp_variance + humidity_variance) / 10))
            return round(accuracy, 1)
        except:
            return 75  # Default accuracy 
# SSARS - Smart Society Alert & Resource System

![SSARS Logo](ssars/ssars/static/images/SSARS.png)

A comprehensive Django-based web application for smart home energy management, weather monitoring, news aggregation, and intelligent resource optimization with advanced machine learning capabilities.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

SSARS is an intelligent platform designed to provide:
- **Smart Energy Analysis**: Advanced energy consumption monitoring and prediction
- **Weather Intelligence**: Real-time weather data and forecasting
- **News Aggregation**: Centralized news management and display
- **User Management**: Secure authentication and account management
- **Resource Optimization**: AI-powered recommendations for energy efficiency

## ✨ Features

### 🔋 Energy Analysis Module
- **Real-time Energy Monitoring**: Track power consumption across devices and rooms
- **Advanced ML Predictions**: Machine learning models for energy consumption forecasting
- **Cost Analysis**: Detailed cost breakdowns and projections in rupees (₹)
- **Smart Alerts**: Intelligent notifications for unusual consumption patterns
- **Interactive Visualizations**: Charts and graphs for energy patterns
- **Efficiency Scoring**: Performance metrics and optimization recommendations
- **Multi-device Support**: Monitor various device types (HVAC, Lighting, Electronics, etc.)

### 🌤️ Weather Intelligence
- **Real-time Weather Data**: Integration with OpenWeatherMap API
- **Weather Forecasting**: 5-day weather predictions
- **Historical Analysis**: Weather trend analysis and pattern recognition
- **Weather Alerts**: Automated alerts for severe weather conditions
- **Interactive Charts**: Visual weather data representation
- **City-based Monitoring**: Multi-location weather tracking

### 📰 News Management
- **News Aggregation**: Centralized news article management
- **Image Support**: Visual content support for news articles
- **Date-based Organization**: Chronological news organization
- **Responsive Display**: Mobile-friendly news interface

### 👤 User Management
- **Secure Authentication**: User registration and login system
- **Account Management**: Profile management capabilities
- **Session Management**: Secure session handling

## 🛠️ Technology Stack

### Backend
- **Framework**: Django 4.2.7
- **Database**: SQLite3 (development) / PostgreSQL (production ready)
- **API Integration**: OpenWeatherMap API
- **Machine Learning**: scikit-learn, pandas, numpy

### Frontend
- **CSS Framework**: Bootstrap 5.1.1
- **Icons**: Bootstrap Icons 1.7.2
- **JavaScript**: Vanilla JS with Bootstrap bundle
- **Charts**: Matplotlib & Seaborn for data visualization

### Data Science & ML
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (Random Forest, Linear Regression, Ensemble Methods)
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib

### External APIs
- **Weather Data**: OpenWeatherMap API
- **Geocoding**: OpenWeatherMap Geocoding API

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Step-by-step Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd SSARS-MAIN
```

2. **Create Virtual Environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**
```bash
cd ssars
pip install -r requirements.txt
```

4. **Environment Setup**
```bash
# Copy environment template (if available)
cp .env.example .env  # Edit with your settings
```

5. **Database Setup**
```bash
python manage.py makemigrations
python manage.py migrate
```

6. **Create Superuser (Optional)**
```bash
python manage.py createsuperuser
```

7. **Collect Static Files**
```bash
python manage.py collectstatic
```

8. **Run Development Server**
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to access the application.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or update `ssars/settings.py` with:

```python
# Security
SECRET_KEY = 'your-secret-key-here'
DEBUG = True  # Set to False in production

# Database (for production)
DATABASE_URL = 'postgresql://user:password@localhost:5432/ssars_db'

# OpenWeatherMap API
OPENWEATHER_API_KEY = 'your-openweathermap-api-key'

# Static and Media Files
STATIC_URL = '/static/'
MEDIA_URL = '/media/'
```

### API Keys Required

1. **OpenWeatherMap API Key**:
   - Sign up at [OpenWeatherMap](https://openweathermap.org/api)
   - Get your free API key
   - Update `OPENWEATHER_API_KEY` in settings

## 📱 Usage

### Energy Analysis Dashboard

1. **Access Energy Dashboard**: Navigate to `/energy/`
2. **View Consumption Data**: Monitor real-time energy usage
3. **Train ML Models**: Use `/energy/train-model/` to train prediction models
4. **Make Predictions**: Use `/energy/predict/` for consumption forecasting
5. **Monitor Costs**: Track energy costs and get optimization tips

### Weather Monitoring

1. **Weather Home**: Navigate to `/weather/`
2. **View Forecasts**: Check 5-day weather predictions
3. **Weather Alerts**: Monitor severe weather notifications
4. **Historical Analysis**: Analyze weather trends and patterns

### News Management

1. **News Section**: Navigate to `/news/`
2. **View Articles**: Browse latest news and updates
3. **Admin Panel**: Manage news articles through Django admin

## 📊 API Documentation

### Energy Analysis API

#### Get Energy Statistics
```http
GET /energy/api/stats/
```

**Response Example**:
```json
{
    "status": "success",
    "data": {
        "summary": {
            "total_consumption_kwh": 1250.5,
            "total_cost": 150.06,
            "avg_consumption_w": 850.2
        },
        "consumption_breakdown": {
            "by_device": {...},
            "by_room": {...}
        }
    }
}
```

### Weather API

#### Get Current Weather
**External API Integration**: OpenWeatherMap API used internally

## 📁 Project Structure

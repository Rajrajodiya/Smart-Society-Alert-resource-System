from django.db import models
from django.utils import timezone
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnergyData(models.Model):
    home_id = models.IntegerField()
    timestamp = models.DateTimeField()
    device_type = models.CharField(max_length=100)
    device_name = models.CharField(max_length=100)
    room = models.CharField(max_length=100)
    status = models.CharField(max_length=50)
    power_watt = models.FloatField()
    user_present = models.BooleanField()
    activity = models.CharField(max_length=100)
    indoor_temp = models.FloatField()
    outdoor_temp = models.FloatField()
    humidity = models.FloatField()
    light_level = models.FloatField()
    day_of_week = models.IntegerField()
    hour_of_day = models.IntegerField()
    price_kwh = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Home {self.home_id} - {self.device_name} - {self.timestamp}"

class EnergyAlert(models.Model):
    ALERT_TYPES = [
        ('HIGH_CONSUMPTION', 'High Consumption'),
        ('UNUSUAL_PATTERN', 'Unusual Pattern'),
        ('EFFICIENCY_LOSS', 'Efficiency Loss'),
        ('COST_WARNING', 'Cost Warning'),
        ('MAINTENANCE_DUE', 'Maintenance Due'),
    ]
    
    SEVERITY_LEVELS = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]
    
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS)
    message = models.TextField()
    home_id = models.IntegerField()
    device_name = models.CharField(max_length=100, blank=True)
    threshold_value = models.FloatField()
    actual_value = models.FloatField()
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_alert_type_display()} - {self.severity} - Home {self.home_id}"

class EnergyModel(models.Model):
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50)
    model_file = models.FileField(upload_to='models/')
    accuracy = models.FloatField()
    mse = models.FloatField()
    mae = models.FloatField()
    r2_score = models.FloatField()
    training_data_size = models.IntegerField()
    feature_count = models.IntegerField()
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.model_type} (R²: {self.r2_score:.4f})"

class AdvancedEnergyAnalysis:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.label_encoders = {}
        self.feature_names = []
        self.model_metrics = {}
        self.is_trained = False
        
    def validate_input_data(self, features):
        """Validate input data for predictions"""
        errors = []
        
        # Temperature validation
        if 'indoor_temp' in features:
            if not isinstance(features['indoor_temp'], (int, float)):
                errors.append("Indoor temperature must be a number")
            elif features['indoor_temp'] < -50 or features['indoor_temp'] > 100:
                errors.append("Indoor temperature must be between -50°C and 100°C")
        
        if 'outdoor_temp' in features:
            if not isinstance(features['outdoor_temp'], (int, float)):
                errors.append("Outdoor temperature must be a number")
            elif features['outdoor_temp'] < -50 or features['outdoor_temp'] > 100:
                errors.append("Outdoor temperature must be between -50°C and 100°C")
        
        # Humidity validation
        if 'humidity' in features:
            if not isinstance(features['humidity'], (int, float)):
                errors.append("Humidity must be a number")
            elif features['humidity'] < 0 or features['humidity'] > 100:
                errors.append("Humidity must be between 0% and 100%")
        
        # Light level validation
        if 'light_level' in features:
            if not isinstance(features['light_level'], (int, float)):
                errors.append("Light level must be a number")
            elif features['light_level'] < 0 or features['light_level'] > 50000:
                errors.append("Light level must be between 0 and 50,000 lux")
        
        # Time validation
        if 'hour_of_day' in features:
            if not isinstance(features['hour_of_day'], int):
                errors.append("Hour of day must be an integer")
            elif features['hour_of_day'] < 0 or features['hour_of_day'] > 23:
                errors.append("Hour of day must be between 0 and 23")
        
        if 'day_of_week' in features:
            if not isinstance(features['day_of_week'], int):
                errors.append("Day of week must be an integer")
            elif features['day_of_week'] < 0 or features['day_of_week'] > 6:
                errors.append("Day of week must be between 0 and 6")
        
        # Price validation
        if 'price_kwh' in features:
            if not isinstance(features['price_kwh'], (int, float)):
                errors.append("Price per kWh must be a number")
            elif features['price_kwh'] < 0 or features['price_kwh'] > 10:
                errors.append("Price per kWh must be between 0 and 10")
        
        return errors
    
    def load_and_prepare_data(self):
        """Load and prepare energy data with advanced feature engineering"""
        try:
            # Try multiple possible paths for the CSV file
            possible_paths = [
                'ssars/static/data/Smart Home Energy Consumption Optimization.csv',
                'static/data/Smart Home Energy Consumption Optimization.csv',
                'ssars/ssars/static/data/Smart Home Energy Consumption Optimization.csv',
            ]
            
            df = None
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Loading energy data from: {path}")
                    # Read in chunks to handle large files
                    chunk_list = []
                    chunk_size = 10000
                    for chunk in pd.read_csv(path, chunksize=chunk_size):
                        chunk_list.append(chunk)
                        if len(chunk_list) >= 100:  # Limit to first 1M rows
                            break
                    df = pd.concat(chunk_list, ignore_index=True)
                    print(f"Loaded {len(df)} records")
                    break
            
            if df is None:
                print("Energy data CSV not found")
                return None
            
            # Clean and prepare data
            original_size = len(df)
            
            # Handle timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Remove rows with missing critical data
            df = df.dropna(subset=['power_watt'])
            print(f"Removed {original_size - len(df)} rows with missing power data")
            
            # Extract advanced features
            df = self.extract_advanced_features(df)
            df = self.encode_categorical_features(df)
            df = self.create_interaction_features(df)
            df = self.remove_outliers(df)
            
            print(f"Final prepared data shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error loading and preparing data: {e}")
            return None
    
    def extract_advanced_features(self, df):
        """Extract advanced time and environmental features"""
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            df['quarter'] = df['timestamp'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_holiday'] = ((df['month'] == 12) & (df['day'] >= 24)).astype(int)  # Simple holiday detection
            
            # Advanced time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        else:
            # Use existing hour_of_day and day_of_week if available
            if 'hour_of_day' in df.columns:
                df['hour'] = df['hour_of_day']
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            if 'day_of_week' in df.columns:
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Season detection
        if 'month' in df.columns:
            df['season'] = ((df['month'] % 12 + 3) // 3).map({
                1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'
            })
        
        # Time period categorization
        if 'hour' in df.columns:
            df['time_period'] = pd.cut(df['hour'], 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                     include_lowest=True)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_peak_hour'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
        
        # Environmental features
        if 'indoor_temp' in df.columns and 'outdoor_temp' in df.columns:
            df['temp_difference'] = df['indoor_temp'] - df['outdoor_temp']
            df['temp_ratio'] = df['indoor_temp'] / (df['outdoor_temp'] + 1e-6)
            df['cooling_need'] = np.maximum(0, df['indoor_temp'] - 22)
            df['heating_need'] = np.maximum(0, 18 - df['indoor_temp'])
        
        # Comfort and efficiency features
        if 'indoor_temp' in df.columns and 'humidity' in df.columns:
            df['comfort_index'] = self.calculate_comfort_index(df['indoor_temp'], df['humidity'])
            df['heat_index'] = self.calculate_heat_index(df['indoor_temp'], df['humidity'])
        
        # Light-based features
        if 'light_level' in df.columns:
            df['light_category'] = pd.cut(df['light_level'], 
                                        bins=[0, 100, 500, 1000, float('inf')], 
                                        labels=['Dark', 'Dim', 'Normal', 'Bright'])
            df['artificial_light_need'] = (df['light_level'] < 300).astype(int)
        
        # Device efficiency features
        if 'device_type' in df.columns:
            # Group devices by efficiency class
            high_efficiency_devices = ['LED', 'ENERGY_STAR', 'SMART']
            df['is_efficient_device'] = df['device_type'].str.contains('|'.join(high_efficiency_devices), 
                                                                     case=False, na=False).astype(int)
        
        # Activity-based features
        if 'activity' in df.columns:
            high_energy_activities = ['cooking', 'washing', 'drying', 'gaming']
            df['high_energy_activity'] = df['activity'].str.contains('|'.join(high_energy_activities), 
                                                                   case=False, na=False).astype(int)
        
        return df
    
    def calculate_comfort_index(self, temp, humidity):
        """Calculate thermal comfort index (0-100 scale)"""
        # Optimal comfort: 20-24°C, 40-60% humidity
        temp_comfort = 100 - np.abs(temp - 22) * 5
        humidity_comfort = 100 - np.abs(humidity - 50) * 2
        return np.clip((temp_comfort + humidity_comfort) / 2, 0, 100)
    
    def calculate_heat_index(self, temp, humidity):
        """Calculate heat index (feels-like temperature)"""
        # Simplified heat index calculation
        hi = temp + 0.5 * (temp + 61.0) + ((humidity - 10.0) / 25.0) * ((17.0 - np.abs(temp - 25.0)) / 17.0)
        return hi
    
    def encode_categorical_features(self, df):
        """Advanced categorical feature encoding"""
        categorical_columns = ['device_type', 'room', 'status', 'activity']
        
        # Add device_name if it exists
        if 'device_name' in df.columns:
            categorical_columns.append('device_name')
        
        for col in categorical_columns:
            if col in df.columns:
                # Label encoding
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
                # One-hot encoding for high-cardinality features
                if df[col].nunique() <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
        
        # Season encoding if exists
        if 'season' in df.columns:
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
        
        # Time period encoding
        if 'time_period' in df.columns:
            time_dummies = pd.get_dummies(df['time_period'], prefix='time_period')
            df = pd.concat([df, time_dummies], axis=1)
        
        # Light category encoding
        if 'light_category' in df.columns:
            light_dummies = pd.get_dummies(df['light_category'], prefix='light')
            df = pd.concat([df, light_dummies], axis=1)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        # Temperature interactions
        if 'indoor_temp' in df.columns and 'outdoor_temp' in df.columns:
            if 'user_present' in df.columns:
                df['temp_user_interaction'] = df['temp_difference'] * df['user_present']
        
        # Light and time interactions
        if 'light_level' in df.columns and 'hour' in df.columns:
            df['light_hour_interaction'] = df['light_level'] * df['hour']
        
        # Humidity and temperature interactions
        if 'humidity' in df.columns and 'indoor_temp' in df.columns:
            df['humid_temp_interaction'] = df['humidity'] * df['indoor_temp']
        
        # Price and time interactions
        if 'price_kwh' in df.columns and 'hour' in df.columns:
            df['price_hour_interaction'] = df['price_kwh'] * df['hour']
        
        # User presence and device interactions
        if 'user_present' in df.columns and 'device_type_encoded' in df.columns:
            df['user_device_interaction'] = df['user_present'] * df['device_type_encoded']
        
        return df
    
    def remove_outliers(self, df):
        """Remove outliers using multiple methods"""
        original_size = len(df)
        
        # Remove obvious outliers in power consumption
        if 'power_watt' in df.columns:
            Q1 = df['power_watt'].quantile(0.25)
            Q3 = df['power_watt'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['power_watt'] >= lower_bound) & (df['power_watt'] <= upper_bound)]
        
        # Remove outliers in environmental parameters
        for col in ['indoor_temp', 'outdoor_temp', 'humidity', 'light_level']:
            if col in df.columns:
                Q1 = df[col].quantile(0.05)  # More conservative outlier removal
                Q3 = df[col].quantile(0.95)
                df = df[(df[col] >= Q1) & (df[col] <= Q3)]
        
        print(f"Removed {original_size - len(df)} outlier rows")
        return df
    
    def prepare_model_features(self, df):
        """Prepare optimized feature set for machine learning"""
        # Base numerical features
        numerical_features = [
            'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_peak_hour',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos'
        ]
        
        # Add environmental features if available
        environmental_features = [
            'indoor_temp', 'outdoor_temp', 'humidity', 'light_level',
            'temp_difference', 'temp_ratio', 'cooling_need', 'heating_need',
            'comfort_index', 'heat_index', 'artificial_light_need'
        ]
        
        for feature in environmental_features:
            if feature in df.columns:
                numerical_features.append(feature)
        
        # Add interaction features
        interaction_features = [
            'temp_user_interaction', 'light_hour_interaction', 'humid_temp_interaction',
            'price_hour_interaction', 'user_device_interaction'
        ]
        
        for feature in interaction_features:
            if feature in df.columns:
                numerical_features.append(feature)
        
        # Add encoded categorical features
        categorical_encoded = [col for col in df.columns if col.endswith('_encoded')]
        numerical_features.extend(categorical_encoded)
        
        # Add one-hot encoded features
        onehot_features = []
        for prefix in ['device_type', 'room', 'status', 'activity', 'season', 'time_period', 'light']:
            onehot_cols = [col for col in df.columns if col.startswith(f'{prefix}_')]
            onehot_features.extend(onehot_cols)
        
        numerical_features.extend(onehot_features)
        
        # Add other important features
        other_features = ['user_present', 'price_kwh', 'is_efficient_device', 'high_energy_activity']
        for feature in other_features:
            if feature in df.columns:
                numerical_features.append(feature)
        
        # Remove duplicates and ensure all features exist
        feature_set = []
        for feature in numerical_features:
            if feature in df.columns and feature not in feature_set:
                feature_set.append(feature)
        
        return df, feature_set
    
    def create_ensemble_model(self):
        """Create advanced ensemble model with multiple algorithms"""
        # Define base models
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        ridge = Ridge(alpha=1.0)
        
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        
        # Create ensemble
        ensemble = VotingRegressor([
            ('rf', rf),
            ('ridge', ridge),
            ('elastic', elastic)
        ])
        
        return ensemble
    
    def train_advanced_model(self, df, model_type='ensemble'):
        """Train advanced energy consumption prediction model"""
        df = self.load_and_prepare_data() if df is None else df
        if df is None or df.empty:
            print("No data available for training")
            return False
        
        try:
            print("Training advanced energy prediction model...")
            
            # Prepare features and target
            df, feature_cols = self.prepare_model_features(df)
            
            X = df[feature_cols]
            y = df['power_watt']
            
            print(f"Training with {len(feature_cols)} features and {len(df)} samples")
            
            # Split data with stratification by power consumption ranges
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Feature selection
            selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_cols)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Get selected feature names
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
            # Scale features using robust scaler (less sensitive to outliers)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train ensemble model
            if model_type == 'ensemble':
                model = self.create_ensemble_model()
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
            elif model_type == 'linear_regression':
                model = LinearRegression()
            else:
                model = self.create_ensemble_model()
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate comprehensive metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            # Store model and related objects
            self.models['energy'] = model
            self.scalers['energy'] = scaler
            self.feature_selectors['energy'] = selector
            self.feature_names = selected_features
            
            # Store comprehensive metrics
            self.model_metrics = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'feature_count': len(selected_features),
                'training_samples': len(X_train)
            }
            
            self.is_trained = True
            
            print("Advanced energy prediction model trained successfully!")
            print(f"Model Performance:")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MAE: {test_mae:.4f} W")
            print(f"  Test RMSE: {test_rmse:.4f} W")
            print(f"  Test MAPE: {test_mape:.4f}%")
            print(f"  Cross-validation R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            print(f"  Features used: {len(selected_features)}")
            
            return True
            
        except Exception as e:
            print(f"Error training advanced model: {e}")
            return False
    
    def predict_consumption_advanced(self, features):
        """Make advanced energy consumption predictions with uncertainty quantification"""
        # Validate inputs
        validation_errors = self.validate_input_data(features)
        if validation_errors:
            return {
                'error': True,
                'message': '; '.join(validation_errors)
            }
        
        # Train model if not already trained
        if not self.is_trained:
            if not self.train_advanced_model(None):
                return {
                    'error': True,
                    'message': 'Failed to train prediction model'
                }
        
        try:
            # Create comprehensive feature vector
            feature_dict = self.create_comprehensive_features(features)
            
            # Create feature vector in the same order as training
            feature_vector = []
            for feature_name in self.feature_names:
                value = feature_dict.get(feature_name, 0)
                feature_vector.append(value)
            
            # Apply feature selection
            feature_vector_selected = self.feature_selectors['energy'].transform([feature_vector])
            
            # Scale features
            feature_vector_scaled = self.scalers['energy'].transform(feature_vector_selected)
            
            # Make prediction
            prediction = self.models['energy'].predict(feature_vector_scaled)[0]
            
            # Calculate prediction confidence (for ensemble models)
            if hasattr(self.models['energy'], 'estimators_'):
                # Get predictions from individual models in ensemble
                individual_predictions = []
                for name, estimator in self.models['energy'].estimators_:
                    pred = estimator.predict(feature_vector_scaled)[0]
                    individual_predictions.append(pred)
                
                prediction_std = np.std(individual_predictions)
                confidence_interval = {
                    'lower': prediction - 1.96 * prediction_std,
                    'upper': prediction + 1.96 * prediction_std,
                    'std': prediction_std
                }
            else:
                confidence_interval = {
                    'lower': prediction * 0.9,
                    'upper': prediction * 1.1,
                    'std': prediction * 0.05
                }
            
            # Calculate efficiency score
            efficiency_score = self.calculate_efficiency_score(prediction, features)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(prediction, features, efficiency_score)
            
            return {
                'error': False,
                'prediction': round(prediction, 2),
                'confidence_interval': {
                    'lower': round(confidence_interval['lower'], 2),
                    'upper': round(confidence_interval['upper'], 2),
                    'std': round(confidence_interval['std'], 2)
                },
                'efficiency_score': round(efficiency_score, 2),
                'model_performance': self.model_metrics,
                'input_features': features,
                'derived_features': {
                    'comfort_index': round(feature_dict.get('comfort_index', 50), 2),
                    'temp_difference': round(feature_dict.get('temp_difference', 0), 2),
                    'is_peak_hour': feature_dict.get('is_peak_hour', 0),
                    'cooling_need': round(feature_dict.get('cooling_need', 0), 2),
                    'heating_need': round(feature_dict.get('heating_need', 0), 2)
                },
                'recommendations': recommendations,
                'feature_importance': self.get_feature_importance()
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': f'Prediction error: {str(e)}'
            }
    
    def create_comprehensive_features(self, input_features):
        """Create comprehensive feature dictionary from input features"""
        # Start with current time if not provided
        now = datetime.now()
        
        # Extract time features
        features = {
            'hour': input_features.get('hour_of_day', now.hour),
            'day_of_week': input_features.get('day_of_week', now.weekday()),
            'is_weekend': 1 if input_features.get('day_of_week', now.weekday()) >= 5 else 0,
            'is_night': 1 if input_features.get('hour_of_day', now.hour) in [22, 23, 0, 1, 2, 3, 4, 5, 6] else 0,
            'is_peak_hour': 1 if input_features.get('hour_of_day', now.hour) in [17, 18, 19, 20, 21] else 0,
        }
        
        # Add cyclical time features
        hour = input_features.get('hour_of_day', now.hour)
        month = now.month
        day_of_year = now.timetuple().tm_yday
        
        features.update({
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'day_sin': np.sin(2 * np.pi * day_of_year / 365),
            'day_cos': np.cos(2 * np.pi * day_of_year / 365),
        })
        
        # Add environmental features
        indoor_temp = input_features.get('indoor_temp', 22)
        outdoor_temp = input_features.get('outdoor_temp', 20)
        humidity = input_features.get('humidity', 50)
        light_level = input_features.get('light_level', 500)
        
        features.update({
            'indoor_temp': indoor_temp,
            'outdoor_temp': outdoor_temp,
            'humidity': humidity,
            'light_level': light_level,
            'temp_difference': indoor_temp - outdoor_temp,
            'temp_ratio': indoor_temp / (outdoor_temp + 1e-6),
            'cooling_need': max(0, indoor_temp - 22),
            'heating_need': max(0, 18 - indoor_temp),
            'comfort_index': self.calculate_comfort_index(indoor_temp, humidity),
            'heat_index': self.calculate_heat_index(indoor_temp, humidity),
            'artificial_light_need': 1 if light_level < 300 else 0,
        })
        
        # Add other features
        features.update({
            'user_present': 1 if input_features.get('user_present', True) else 0,
            'price_kwh': input_features.get('price_kwh', 0.12),
        })
        
        # Add interaction features
        features.update({
            'temp_user_interaction': features['temp_difference'] * features['user_present'],
            'light_hour_interaction': light_level * hour,
            'humid_temp_interaction': humidity * indoor_temp,
            'price_hour_interaction': features['price_kwh'] * hour,
        })
        
        return features
    
    def calculate_efficiency_score(self, predicted_consumption, features):
        """Calculate energy efficiency score (0-100)"""
        # Base efficiency calculation
        indoor_temp = features.get('indoor_temp', 22)
        outdoor_temp = features.get('outdoor_temp', 20)
        user_present = features.get('user_present', True)
        hour = features.get('hour_of_day', 12)
        
        # Ideal consumption for given conditions (simplified model)
        ideal_base = 50  # Base load in watts
        
        # Temperature-based consumption
        temp_diff = abs(indoor_temp - outdoor_temp)
        temp_consumption = temp_diff * 10  # 10W per degree difference
        
        # Time-based consumption
        if 6 <= hour <= 22:  # Daytime
            time_consumption = 30
        else:  # Nighttime
            time_consumption = 10
        
        # User presence factor
        user_factor = 1.5 if user_present else 0.5
        
        ideal_consumption = (ideal_base + temp_consumption + time_consumption) * user_factor
        
        # Calculate efficiency score
        if predicted_consumption <= ideal_consumption:
            efficiency_score = 100
        else:
            excess = predicted_consumption - ideal_consumption
            efficiency_score = max(0, 100 - (excess / ideal_consumption) * 100)
        
        return efficiency_score
    
    def generate_recommendations(self, predicted_consumption, features, efficiency_score):
        """Generate energy efficiency recommendations"""
        recommendations = []
        
        # Temperature-based recommendations
        indoor_temp = features.get('indoor_temp', 22)
        outdoor_temp = features.get('outdoor_temp', 20)
        
        if indoor_temp > 25:
            recommendations.append({
                'type': 'temperature',
                'priority': 'high',
                'message': f'Indoor temperature is high ({indoor_temp}°C). Consider reducing by 2-3°C to save energy.',
                'potential_savings': '10-15% reduction in cooling costs'
            })
        elif indoor_temp < 18:
            recommendations.append({
                'type': 'temperature',
                'priority': 'high',
                'message': f'Indoor temperature is low ({indoor_temp}°C). Consider increasing by 2-3°C to save energy.',
                'potential_savings': '10-15% reduction in heating costs'
            })
        
        # Humidity recommendations
        humidity = features.get('humidity', 50)
        if humidity > 70:
            recommendations.append({
                'type': 'humidity',
                'priority': 'medium',
                'message': f'High humidity ({humidity}%) increases cooling load. Consider using a dehumidifier.',
                'potential_savings': '5-10% reduction in cooling costs'
            })
        
        # Time-based recommendations
        hour = features.get('hour_of_day', 12)
        if 17 <= hour <= 21:  # Peak hours
            recommendations.append({
                'type': 'time',
                'priority': 'medium',
                'message': 'Peak electricity hours. Consider shifting non-essential usage to off-peak times.',
                'potential_savings': 'Reduce electricity costs by 20-30%'
            })
        
        # Efficiency-based recommendations
        if efficiency_score < 60:
            recommendations.append({
                'type': 'efficiency',
                'priority': 'high',
                'message': 'Low energy efficiency detected. Consider upgrading to energy-efficient appliances.',
                'potential_savings': '20-40% reduction in overall consumption'
            })
        elif efficiency_score < 80:
            recommendations.append({
                'type': 'efficiency',
                'priority': 'medium',
                'message': 'Good efficiency but room for improvement. Check for phantom loads and optimize usage patterns.',
                'potential_savings': '5-15% potential savings'
            })
        
        # Light-based recommendations
        light_level = features.get('light_level', 500)
        if light_level > 1000 and 6 <= hour <= 18:
            recommendations.append({
                'type': 'lighting',
                'priority': 'low',
                'message': 'High natural light available. Consider turning off artificial lights.',
                'potential_savings': '5-10% reduction in lighting costs'
            })
        
        return recommendations
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained or 'energy' not in self.models:
            return {}
        
        model = self.models['energy']
        
        # For ensemble models, average the feature importance
        if hasattr(model, 'estimators_'):
            importances = []
            for name, estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                avg_importance = np.mean(importances, axis=0)
                return dict(zip(self.feature_names, avg_importance))
        elif hasattr(model, 'feature_importances_'):
            return dict(zip(self.feature_names, model.feature_importances_))
        
        return {}
    
    def save_model(self, filepath):
        """Save trained model and associated objects"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics
        }
        
        joblib.dump(model_data, filepath)
        return True
    
    def load_model(self, filepath):
        """Load trained model and associated objects"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data.get('models', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_selectors = model_data.get('feature_selectors', {})
            self.label_encoders = model_data.get('label_encoders', {})
            self.feature_names = model_data.get('feature_names', [])
            self.model_metrics = model_data.get('model_metrics', {})
            
            self.is_trained = len(self.models) > 0
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Backward compatibility - keep the old class name as an alias
EnergyAnalysis = AdvancedEnergyAnalysis 
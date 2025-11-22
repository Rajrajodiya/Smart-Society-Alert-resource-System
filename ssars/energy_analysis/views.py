from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Avg, Sum, Count
from django.utils import timezone
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import os
from .models import EnergyData, EnergyModel, EnergyAlert, AdvancedEnergyAnalysis
from .forms import PredictionForm
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

def get_csv_path():
    """Get the correct path to the CSV file"""
    possible_paths = [
        'ssars/static/data/Smart Home Energy Consumption Optimization.csv',
        'static/data/Smart Home Energy Consumption Optimization.csv',
        os.path.join('ssars', 'ssars', 'static', 'data', 'Smart Home Energy Consumption Optimization.csv'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                    'ssars', 'static', 'data', 'Smart Home Energy Consumption Optimization.csv'),
        os.path.join(os.path.dirname(__file__), 
                    '..', 'ssars', 'static', 'data', 'Smart Home Energy Consumption Optimization.csv'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def load_energy_data():
    """Load and prepare energy data from CSV with memory optimization"""
    csv_path = get_csv_path()
    
    if csv_path is None:
        return None
    
    try:
        print(f"Loading energy data from: {csv_path}")
        
        # Load data in chunks to manage memory
        chunks = []
        chunk_size = 50000
        total_loaded = 0
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            chunks.append(chunk)
            total_loaded += len(chunk)
            if total_loaded >= 500000:  # Limit to 500k records for performance
                break
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"CSV columns: {list(df.columns)}")
        
        # Clean and prepare data
        print(f"Sample timestamp: {df['timestamp'].iloc[0]}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
        
        print(f"Successfully loaded {len(df)} records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def energy_dashboard(request):
    """Enhanced comprehensive dashboard for energy analysis with visualization, monitoring, and analysis"""
    try:
        df = load_energy_data()
        
        if df is None:
            context = {
                'error': 'Energy data file not found. Please check if the CSV file exists.',
                'has_data': False
            }
            return render(request, 'energy_analysis/dashboard.html', context)
        
        # Calculate enhanced metrics
        df['energy_kwh'] = df['power_watt'] / 1000
        
        # Handle missing price_kwh column
        if 'price_kwh' not in df.columns:
            if 'price_kWh' in df.columns:
                df['price_kwh'] = df['price_kWh']
            elif 'Price_kWh' in df.columns:
                df['price_kwh'] = df['Price_kWh']
            elif 'price per kwh' in df.columns:
                df['price_kwh'] = df['price per kwh']
            else:
                df['price_kwh'] = 0.12
                print("Warning: price_kwh column not found. Using default value of ₹0.12/kWh")
        
        df['cost'] = df['energy_kwh'] * df['price_kwh']
        
        # ========== BASIC STATISTICS ==========
        total_consumption = df['power_watt'].sum()
        total_cost = df['cost'].sum()
        avg_consumption = df['power_watt'].mean()
        peak_consumption = df['power_watt'].max()
        
        # Device and room analytics
        device_consumption = df.groupby('device_type')['power_watt'].sum().sort_values(ascending=False)
        device_costs = df.groupby('device_type')['cost'].sum().sort_values(ascending=False)
        room_consumption = df.groupby('room')['power_watt'].sum().sort_values(ascending=False)
        room_costs = df.groupby('room')['cost'].sum().sort_values(ascending=False)
        
        # Time-based patterns
        hourly_consumption = df.groupby('hour_of_day')['power_watt'].mean()
        daily_consumption = df.groupby('day_of_week')['power_watt'].mean()
        
        # Efficiency metrics
        user_present_avg = df[df['user_present'] == True]['power_watt'].mean()
        user_absent_avg = df[df['user_present'] == False]['power_watt'].mean()
        efficiency_ratio = (1 - user_absent_avg / user_present_avg) * 100
        
        # ========== VISUALIZATION CHARTS ==========
        charts = {}
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        
        # 1. Time Series Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        df['date'] = df['timestamp'].dt.date
        daily_consumption_data = df.groupby('date')['power_watt'].sum()
        daily_consumption_data.tail(30).plot(ax=ax, kind='line', color='blue', linewidth=2)
        ax.set_title('Energy Consumption Over Time (Last 30 Days)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Consumption (W)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['time_series'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 2. Device Consumption Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        device_consumption.head(8).plot(kind='bar', ax=ax, color='skyblue', edgecolor='navy')
        ax.set_title('Total Energy Consumption by Device Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Device Type')
        ax.set_ylabel('Total Consumption (W)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['device_consumption'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 3. Room Consumption Pie Chart
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(room_consumption)))
        ax.pie(room_consumption.values, labels=room_consumption.index, autopct='%1.1f%%', 
               startangle=90, colors=colors)
        ax.set_title('Energy Consumption Distribution by Room', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['room_consumption'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 4. Hourly Pattern Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hourly_consumption.index, hourly_consumption.values, marker='o', linewidth=2, color='red')
        ax.set_title('Average Hourly Energy Consumption Pattern', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Consumption (W)')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['hourly_pattern'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 5. Cost Analysis Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Daily cost trends
        daily_costs = df.groupby('date')['cost'].sum()
        daily_costs.tail(30).plot(ax=ax1, kind='line', color='green', linewidth=2)
        ax1.set_title('Daily Energy Costs (Last 30 Days)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cost (₹)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Device cost breakdown
        device_costs.head(8).plot(ax=ax2, kind='bar', color='orange', edgecolor='red')
        ax2.set_title('Cost by Device Type', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Device Type')
        ax2.set_ylabel('Total Cost (₹)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Hourly cost pattern
        hourly_costs = df.groupby('hour_of_day')['cost'].mean()
        ax3.plot(hourly_costs.index, hourly_costs.values, marker='o', linewidth=2, color='purple')
        ax3.set_title('Average Hourly Energy Costs', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Average Cost (₹)')
        ax3.set_xticks(range(0, 24, 3))
        ax3.grid(True, alpha=0.3)
        
        # Efficiency comparison
        presence_consumption = df.groupby('user_present')['power_watt'].mean()
        presence_labels = ['User Absent', 'User Present']
        ax4.bar(presence_labels, presence_consumption.values, 
               color=['lightcoral', 'lightgreen'], edgecolor='black')
        ax4.set_title('Consumption vs User Presence', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Consumption (W)')
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['monitoring_analysis'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 6. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_cols = ['power_watt', 'indoor_temp', 'outdoor_temp', 'humidity', 'light_level']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        correlation_matrix = df[numeric_cols].corr()
        
        import seaborn as sns
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Between Energy Consumption and Environmental Factors', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['correlation'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # ========== MONITORING STATISTICS ==========
        monitoring_stats = {
            'total_cost': round(total_cost, 2),
            'daily_avg_cost': round(df.groupby('date')['cost'].sum().mean(), 2),
            'monthly_projected_cost': round(df.groupby('date')['cost'].sum().mean() * 30, 2),
            'total_consumption_kwh': round(df['energy_kwh'].sum(), 2),
            'daily_avg_consumption': round(df.groupby('date')['energy_kwh'].sum().mean(), 2),
            'peak_consumption_hour': int(df.groupby('hour_of_day')['power_watt'].mean().idxmax()),
            'most_expensive_device': device_costs.index[0] if len(device_costs) > 0 else 'N/A',
            'most_expensive_room': room_costs.index[0] if len(room_costs) > 0 else 'N/A',
            'efficiency_score': round(efficiency_ratio, 1),
        }
        
        # ========== ANALYSIS INSIGHTS ==========
        # Recent alerts simulation
        recent_alerts = []
        if avg_consumption > 1000:
            recent_alerts.append({
                'type': 'high_consumption',
                'title': 'High Energy Consumption',
                'message': f'Current average consumption ({avg_consumption:.1f}W) is above normal levels',
                'severity': 'warning'
            })
        
        if efficiency_ratio < 50:
            recent_alerts.append({
                'type': 'low_efficiency', 
                'title': 'Low Energy Efficiency',
                'message': f'Energy efficiency is at {efficiency_ratio:.1f}% - consider optimization',
                'severity': 'info'
            })
        
        if total_cost > 100:
            recent_alerts.append({
                'type': 'high_cost',
                'title': 'High Energy Costs',
                'message': f'Total energy costs (₹{total_cost:.2f}) are elevated',
                'severity': 'warning'
            })
        
        # Analysis insights
        insights = [
            {
                'title': 'Peak Consumption Hour',
                'value': f"{df.groupby('hour_of_day')['power_watt'].mean().idxmax()}:00",
                'description': 'Hour with highest average consumption',
                'icon': 'clock'
            },
            {
                'title': 'Most Efficient Room',
                'value': df.groupby('room')['power_watt'].mean().idxmin(),
                'description': 'Room with lowest average consumption',
                'icon': 'home'
            },
            {
                'title': 'Energy Savings Potential',
                'value': f"{((df[df['user_present']==False]['power_watt'].mean() / df['power_watt'].mean()) * 100):.1f}%",
                'description': 'Potential savings by optimizing usage when absent',
                'icon': 'leaf'
            },
            {
                'title': 'Cost Efficiency Leader',
                'value': device_costs.index[-1] if len(device_costs) > 0 else 'N/A',
                'description': 'Most cost-efficient device type',
                'icon': 'dollar-sign'
            }
        ]
        
        # Recent data for table
        recent_data = df.tail(10).to_dict('records')
        
        context = {
            'has_data': True,
            # Basic stats
            'total_consumption': round(total_consumption, 2),
            'total_cost': round(total_cost, 2),
            'avg_consumption': round(avg_consumption, 2),
            'peak_consumption': round(peak_consumption, 2),
            'total_devices': df['device_type'].nunique(),
            'unique_rooms': df['room'].nunique(),
            'data_points': len(df),
            'date_range': f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}",
            
            # Charts for visualization
            'charts': charts,
            
            # Monitoring statistics
            'monitoring_stats': monitoring_stats,
            
            # Analysis data
            'device_consumption': device_consumption.head(8).to_dict(),
            'device_costs': device_costs.head(8).to_dict(),
            'room_consumption': room_consumption.head(8).to_dict(),
            'room_costs': room_costs.head(8).to_dict(),
            'hourly_consumption': hourly_consumption.to_dict(),
            'daily_consumption': daily_consumption.to_dict(),
            'efficiency_ratio': round(efficiency_ratio, 1),
            'user_present_avg': round(user_present_avg, 2),
            'user_absent_avg': round(user_absent_avg, 2),
            
            # Alerts and insights
            'recent_alerts': recent_alerts,
            'insights': insights,
            'recent_data': recent_data,
            
            # Additional metrics
            'peak_hours': df.groupby('hour_of_day')['power_watt'].mean().nlargest(3).to_dict(),
            'off_peak_hours': df.groupby('hour_of_day')['power_watt'].mean().nsmallest(3).to_dict(),
        }
            
    except Exception as e:
        context = {
            'error': f'Error loading comprehensive dashboard: {str(e)}',
            'has_data': False
        }
    
    return render(request, 'energy_analysis/dashboard.html', context)

def energy_data_list(request):
    """Display paginated energy data"""
    try:
        df = load_energy_data()
        
        if df is not None and len(df) > 0:
            # Convert DataFrame to list of dictionaries for template
            data = df.head(1000).to_dict('records')  # Limit to 1000 records for performance
            
            # Pagination
            paginator = Paginator(data, 50)  # Show 50 records per page
            page_number = request.GET.get('page')
            page_obj = paginator.get_page(page_number)
            
            context = {
                'page_obj': page_obj,
                'total_records': len(df),
                'columns': df.columns.tolist()
            }
        else:
            context = {
                'error': 'No energy data available.'
            }
            
    except Exception as e:
        context = {
            'error': f'Error loading data: {str(e)}'
        }
    
    return render(request, 'energy_analysis/data_list.html', context)

def train_models(request):
    """Train energy consumption prediction models"""
    if request.method == 'POST':
        try:
            analyzer = AdvancedEnergyAnalysis()
            df = analyzer.load_and_prepare_data()
            
            if df is not None and len(df) > 0:
                # Extract features
                df = analyzer.extract_advanced_features(df)
                
                # Train models
                model_type = request.POST.get('model_type', 'ensemble')
                results = analyzer.train_advanced_model(df, model_type)
                
                if results:
                    # Save model information to database
                    energy_model = EnergyModel.objects.create(
                        name=f"Energy Model {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        model_type=model_type,
                        accuracy=results.get('accuracy', 0),
                        mse=results.get('mse', 0),
                        mae=results.get('mae', 0),
                        r2_score=results.get('r2_score', 0),
                        training_data_size=len(df),
                        feature_count=len(analyzer.feature_names),
                        is_active=True
                    )
                    
                    messages.success(request, f'Model trained successfully! R² Score: {results.get("r2_score", 0):.4f}')
                else:
                    messages.error(request, 'Failed to train model.')
            else:
                messages.error(request, 'No data available for training.')
                
        except Exception as e:
            messages.error(request, f'Error training models: {str(e)}')
            
        return redirect('energy_analysis:train_model')
    
    # GET request - show training form
    recent_models = EnergyModel.objects.order_by('-created_at')[:10]
    
    context = {
        'recent_models': recent_models
    }
    
    return render(request, 'energy_analysis/train_model.html', context)

def predict_energy_consumption(request):
    """Predict energy consumption based on input parameters"""
    prediction_result = None
    prediction_kwh = None
    prediction_cost = None
    error_message = None
    
    # Get available options for dropdowns first
    device_types = ['Appliance', 'HVAC', 'Lighting', 'Electronics', 'Kitchen']
    rooms = ['Living Room', 'Bedroom', 'Kitchen', 'Bathroom', 'Office']
    
    try:
        df = load_energy_data()
        if df is not None:
            device_types = df['device_type'].unique().tolist()
            rooms = df['room'].unique().tolist()
    except:
        pass  # Use default values
    
    if request.method == 'POST':
        form = PredictionForm(request.POST, device_types=device_types, rooms=rooms)
        
        if form.is_valid():
            try:
                # Get cleaned data from form
                features = {
                    'indoor_temp': form.cleaned_data['indoor_temp'],
                    'outdoor_temp': form.cleaned_data['outdoor_temp'],
                    'humidity': form.cleaned_data['humidity'],
                    'light_level': form.cleaned_data['light_level'],
                    'hour_of_day': form.cleaned_data['hour_of_day'],
                    'day_of_week': form.cleaned_data['day_of_week'],
                    'user_present': form.cleaned_data['user_present'],
                    'device_type': form.cleaned_data['device_type'],
                    'room': form.cleaned_data['room'],
                    'price_kwh': form.cleaned_data['price_kwh']
                }
                
                # Initialize analyzer and make prediction
                analyzer = AdvancedEnergyAnalysis()
                
                # Load and prepare training data if not already done
                if not analyzer.is_trained:
                    df = analyzer.load_and_prepare_data()
                    if df is not None:
                        df = analyzer.extract_advanced_features(df)
                        analyzer.train_advanced_model(df, 'ensemble')
                
                prediction_result = analyzer.predict_consumption_advanced(features)
                
                # Calculate additional values if prediction was successful
                if prediction_result:
                    # Convert watts to kWh
                    prediction_kwh = prediction_result * 0.001
                    # Calculate cost per hour
                    prediction_cost = prediction_kwh * form.cleaned_data['price_kwh']
                
            except Exception as e:
                error_message = f"Prediction error: {str(e)}"
        else:
            error_message = "Please correct the errors in the form"
    else:
        # GET request - create empty form
        form = PredictionForm(device_types=device_types, rooms=rooms)
    
    context = {
        'form': form,
        'prediction_result': prediction_result,
        'prediction_kwh': prediction_kwh,
        'prediction_cost': prediction_cost,
        'error_message': error_message,
        'device_types': device_types,
        'rooms': rooms
    }
    
    return render(request, 'energy_analysis/predict.html', context)

def energy_visualization(request):
    """Generate comprehensive energy consumption visualizations"""
    try:
        df = load_energy_data()
        
        if df is None:
            context = {
                'has_data': False,
                'error': 'Energy data file not found.'
            }
            return render(request, 'energy_analysis/visualization.html', context)
        
        # Handle missing price_kwh column
        if 'price_kwh' not in df.columns:
            if 'price_kWh' in df.columns:
                df['price_kwh'] = df['price_kWh']
            elif 'Price_kWh' in df.columns:
                df['price_kwh'] = df['Price_kWh']
            elif 'price per kwh' in df.columns:
                df['price_kwh'] = df['price per kwh']
            else:
                df['price_kwh'] = 0.12  # Default price per kWh
        
        df['cost'] = df['power_watt'] / 1000 * df['price_kwh']
        
        charts = {}
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        
        # 1. Time Series Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        df['date'] = df['timestamp'].dt.date
        daily_consumption = df.groupby('date')['power_watt'].sum()
        daily_consumption.tail(30).plot(ax=ax, kind='line', color='blue', linewidth=2)
        ax.set_title('Energy Consumption Over Time (Last 30 Days)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Consumption (W)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['time_series'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 2. Device Consumption Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        device_consumption = df.groupby('device_type')['power_watt'].sum().sort_values(ascending=False)
        device_consumption.plot(kind='bar', ax=ax, color='skyblue', edgecolor='navy')
        ax.set_title('Total Energy Consumption by Device Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Device Type')
        ax.set_ylabel('Total Consumption (W)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['device_consumption'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 3. Room Consumption Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        room_consumption = df.groupby('room')['power_watt'].sum().sort_values(ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(room_consumption)))
        ax.pie(room_consumption.values, labels=room_consumption.index, autopct='%1.1f%%', 
               startangle=90, colors=colors)
        ax.set_title('Energy Consumption Distribution by Room', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['room_consumption'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 4. Hourly Pattern Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        hourly_consumption = df.groupby('hour_of_day')['power_watt'].mean()
        ax.plot(hourly_consumption.index, hourly_consumption.values, marker='o', linewidth=2, color='red')
        ax.set_title('Average Hourly Energy Consumption', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Consumption (W)')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['hourly_pattern'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 5. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_cols = ['power_watt', 'indoor_temp', 'outdoor_temp', 'humidity', 'light_level']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        correlation_matrix = df[numeric_cols].corr()
        
        import seaborn as sns
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Between Energy Consumption and Environmental Factors', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['correlation'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 6. Cost Analysis Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        daily_costs = df.groupby('date')['cost'].sum()
        daily_costs.tail(30).plot(ax=ax, kind='line', color='green', linewidth=2)
        ax.set_title('Daily Energy Costs (Last 30 Days)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cost (₹)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['cost_analysis'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Calculate comprehensive statistics
        stats = {
            'total_records': len(df),
            'total_consumption': round(df['power_watt'].sum(), 2),
            'avg_consumption': round(df['power_watt'].mean(), 2),
            'max_consumption': round(df['power_watt'].max(), 2),
            'min_consumption': round(df['power_watt'].min(), 2),
            'unique_devices': df['device_type'].nunique(),
            'unique_rooms': df['room'].nunique(),
            'unique_homes': df['home_id'].nunique(),
            'date_range': f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}",
            'peak_hour': int(df.groupby('hour_of_day')['power_watt'].mean().idxmax()),
            'off_peak_hour': int(df.groupby('hour_of_day')['power_watt'].mean().idxmin()),
            'total_cost': round(df['cost'].sum(), 2),
            'avg_daily_cost': round(df.groupby('date')['cost'].sum().mean(), 2),
        }
        
        context = {
            'charts': charts,
            'stats': stats,
            'has_data': True
        }
        
    except Exception as e:
        context = {
            'has_data': False,
            'error': f'Error creating visualizations: {str(e)}'
        }
    
    return render(request, 'energy_analysis/visualization.html', context)

def utility_monitor(request):
    """Comprehensive utility monitoring with cost analysis and insights"""
    try:
        df = load_energy_data()
        
        if df is None:
            context = {
                'has_data': False,
                'error': 'Energy data file not found or could not be loaded.'
            }
            return render(request, 'energy_analysis/utility_monitor.html', context)
        
        # Calculate basic metrics
        df['energy_kwh'] = df['power_watt'] / 1000  # Convert to kWh
        
        # Handle missing price_kwh column
        if 'price_kwh' not in df.columns:
            if 'price_kWh' in df.columns:
                df['price_kwh'] = df['price_kWh']
            elif 'Price_kWh' in df.columns:
                df['price_kwh'] = df['Price_kWh']
            elif 'price per kwh' in df.columns:
                df['price_kwh'] = df['price per kwh']
            else:
                df['price_kwh'] = 0.12  # Default price per kWh
                print("Warning: price_kwh column not found. Using default value of ₹0.12/kWh")
        
        df['cost'] = df['energy_kwh'] * df['price_kwh']
        
        # Create visualizations
        charts = {}
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        
        # 1. Cost Analysis Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Daily cost trends
        df['date'] = df['timestamp'].dt.date
        daily_costs = df.groupby('date')['cost'].sum()
        daily_costs.tail(30).plot(ax=ax1, kind='line', color='red', linewidth=2)
        ax1.set_title('Daily Energy Costs (Last 30 Days)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cost (₹)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Device cost breakdown
        device_costs = df.groupby('device_type')['cost'].sum().sort_values(ascending=False).head(10)
        device_costs.plot(ax=ax2, kind='bar', color='skyblue', edgecolor='navy')
        ax2.set_title('Cost by Device Type', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Device Type')
        ax2.set_ylabel('Total Cost (₹)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Hourly cost pattern
        hourly_costs = df.groupby('hour_of_day')['cost'].mean()
        ax3.plot(hourly_costs.index, hourly_costs.values, marker='o', linewidth=2, color='green')
        ax3.set_title('Average Hourly Energy Costs', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Average Cost (₹)')
        ax3.set_xticks(range(0, 24, 3))
        ax3.grid(True, alpha=0.3)
        
        # Room cost distribution
        room_costs = df.groupby('room')['cost'].sum().sort_values(ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(room_costs)))
        ax4.pie(room_costs.values, labels=room_costs.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax4.set_title('Cost Distribution by Room', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['cost_analysis'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 2. Energy Consumption Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Daily consumption trends
        daily_consumption = df.groupby('date')['power_watt'].sum() / 1000  # Convert to kWh
        daily_consumption.tail(30).plot(ax=ax1, kind='line', color='blue', linewidth=2)
        ax1.set_title('Daily Energy Consumption (Last 30 Days)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Consumption (kWh)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Device consumption breakdown
        device_consumption = df.groupby('device_type')['power_watt'].sum().sort_values(ascending=False).head(10)
        device_consumption.plot(ax=ax2, kind='bar', color='orange', edgecolor='red')
        ax2.set_title('Consumption by Device Type', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Device Type')
        ax2.set_ylabel('Total Consumption (W)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Hourly consumption heatmap
        hourly_room_consumption = df.groupby(['hour_of_day', 'room'])['power_watt'].mean().unstack()
        im = ax3.imshow(hourly_room_consumption.T, aspect='auto', cmap='YlOrRd')
        ax3.set_title('Consumption Heatmap: Hour vs Room', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Room')
        ax3.set_xticks(range(0, 24, 3))
        ax3.set_xticklabels(range(0, 24, 3))
        ax3.set_yticks(range(len(hourly_room_consumption.columns)))
        ax3.set_yticklabels(hourly_room_consumption.columns, rotation=0)
        plt.colorbar(im, ax=ax3, label='Avg Power (W)')
        
        # Weekly pattern
        weekly_consumption = df.groupby('day_of_week')['power_watt'].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax4.bar(day_names, weekly_consumption.values, color='purple', alpha=0.7)
        ax4.set_title('Average Consumption by Day of Week', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Day of Week')
        ax4.set_ylabel('Average Consumption (W)')
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['consumption_analysis'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 3. Efficiency Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # User presence vs consumption
        presence_consumption = df.groupby('user_present')['power_watt'].mean()
        presence_labels = ['User Absent', 'User Present']
        ax1.bar(presence_labels, presence_consumption.values, 
               color=['lightcoral', 'lightgreen'], edgecolor='black')
        ax1.set_title('Energy Consumption vs User Presence', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Consumption (W)')
        
        # Temperature vs consumption correlation
        temp_consumption = df.groupby(pd.cut(df['indoor_temp'], bins=10))['power_watt'].mean()
        temp_ranges = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in temp_consumption.index]
        ax2.plot(range(len(temp_ranges)), temp_consumption.values, marker='o', color='red')
        ax2.set_title('Consumption vs Indoor Temperature', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Temperature Range (°C)')
        ax2.set_ylabel('Average Consumption (W)')
        ax2.set_xticks(range(len(temp_ranges)))
        ax2.set_xticklabels(temp_ranges, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Device status efficiency
        status_efficiency = df.groupby(['device_type', 'status'])['power_watt'].mean().unstack()
        status_efficiency.plot(ax=ax3, kind='bar', stacked=True)
        ax3.set_title('Device Efficiency by Status', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Device Type')
        ax3.set_ylabel('Average Consumption (W)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Cost efficiency by room
        room_efficiency = df.groupby('room').agg({
            'cost': 'sum',
            'power_watt': 'sum'
        })
        room_efficiency['cost_per_watt'] = room_efficiency['cost'] / room_efficiency['power_watt']
        room_efficiency['cost_per_watt'].sort_values(ascending=False).plot(ax=ax4, kind='bar', color='gold')
        ax4.set_title('Cost Efficiency by Room (₹/W)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Room')
        ax4.set_ylabel('Cost per Watt (₹/W)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        charts['efficiency_analysis'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Calculate comprehensive statistics
        stats = {
            'total_cost': round(df['cost'].sum(), 2),
            'daily_avg_cost': round(df.groupby('date')['cost'].sum().mean(), 2),
            'monthly_projected_cost': round(df.groupby('date')['cost'].sum().mean() * 30, 2),
            'total_consumption_kwh': round(df['energy_kwh'].sum(), 2),
            'daily_avg_consumption': round(df.groupby('date')['energy_kwh'].sum().mean(), 2),
            'peak_consumption_hour': int(df.groupby('hour_of_day')['power_watt'].mean().idxmax()),
            'most_expensive_device': device_costs.index[0],
            'most_expensive_room': room_costs.index[0],
            'efficiency_score': round(100 - (df[df['user_present'] == False]['power_watt'].mean() / df['power_watt'].mean() * 100), 1),
            'total_records': len(df),
            'unique_devices': df['device_type'].nunique(),
            'unique_rooms': df['room'].nunique(),
            'date_range': f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
        }
        
        # Device and room breakdowns
        device_stats = df.groupby('device_type').agg({
            'cost': 'sum',
            'power_watt': ['sum', 'mean'],
            'device_id': 'nunique'
        }).round(2)
        
        room_stats = df.groupby('room').agg({
            'cost': 'sum',
            'power_watt': ['sum', 'mean'],
            'device_id': 'nunique'
        }).round(2)
        
        context = {
            'charts': charts,
            'stats': stats,
            'device_stats': device_stats.to_dict('index'),
            'room_stats': room_stats.to_dict('index'),
            'has_data': True
        }
        
    except Exception as e:
        context = {
            'has_data': False,
            'error': f'Error loading utility monitor: {str(e)}'
        }
    
    return render(request, 'energy_analysis/utility_monitor.html', context)

def energy_alerts(request):
    """Generate intelligent energy consumption alerts"""
    try:
        # Get existing alerts from database
        alerts = EnergyAlert.objects.filter(is_active=True).order_by('-created_at')
        
        # Load data for analysis
        df = load_energy_data()
        
        new_alerts = []
        insights = []
        
        if df is not None and len(df) > 0:
            # Calculate metrics for alerts
            avg_consumption = df['power_watt'].mean()
            current_consumption = df['power_watt'].tail(1000).mean()  # Last 1000 readings
            max_consumption = df['power_watt'].max()
            
            # High consumption alert
            if current_consumption > avg_consumption * 1.3:
                new_alerts.append({
                    'type': 'high_consumption',
                    'title': 'High Energy Consumption Detected',
                    'message': f'Current consumption ({current_consumption:.1f}W) is {((current_consumption/avg_consumption-1)*100):.1f}% above average',
                    'severity': 'warning',
                    'recommendation': 'Check for devices left on unnecessarily'
                })
            
            # Peak hour usage
            current_hour = datetime.now().hour
            peak_hours = df.groupby('hour_of_day')['power_watt'].mean().nlargest(3).index.tolist()
            
            if current_hour in peak_hours:
                new_alerts.append({
                    'type': 'peak_hour',
                    'title': 'Peak Hour Usage',
                    'message': f'Current hour ({current_hour}:00) is a peak consumption period',
                    'severity': 'info',
                    'recommendation': 'Consider shifting non-essential activities to off-peak hours'
                })
            
            # Inefficient device usage
            device_efficiency = df.groupby('device_type').agg({
                'power_watt': 'mean',
                'user_present': 'mean'
            })
            
            for device, data in device_efficiency.iterrows():
                if data['user_present'] < 0.3 and data['power_watt'] > avg_consumption * 0.5:
                    new_alerts.append({
                        'type': 'inefficient_usage',
                        'title': f'Inefficient {device} Usage',
                        'message': f'{device} consuming {data["power_watt"]:.1f}W with low user presence',
                        'severity': 'medium',
                        'recommendation': f'Consider scheduling or automating {device} usage'
                    })
            
            # Generate insights
            insights = [
                {
                    'title': 'Peak Consumption Hour',
                    'value': f"{df.groupby('hour_of_day')['power_watt'].mean().idxmax()}:00",
                    'description': 'Hour with highest average consumption'
                },
                {
                    'title': 'Most Efficient Room',
                    'value': df.groupby('room')['power_watt'].mean().idxmin(),
                    'description': 'Room with lowest average consumption'
                },
                {
                    'title': 'Energy Savings Potential',
                    'value': f"{((df[df['user_present']==False]['power_watt'].mean() / df['power_watt'].mean()) * 100):.1f}%",
                    'description': 'Potential savings by optimizing usage when absent'
                },
                {
                    'title': 'Temperature Sweet Spot',
                    'value': f"{df.loc[df.groupby(pd.cut(df['indoor_temp'], bins=10))['power_watt'].mean().idxmin(), 'indoor_temp']:.1f}°C",
                    'description': 'Indoor temperature with lowest consumption'
                }
            ]
        
        context = {
            'alerts': alerts,
            'new_alerts': new_alerts,
            'insights': insights,
            'total_alerts': alerts.count(),
            'has_data': df is not None and len(df) > 0
        }
        
    except Exception as e:
        context = {
            'error': f'Error loading alerts: {str(e)}',
            'alerts': [],
            'new_alerts': [],
            'insights': [],
            'has_data': False
        }
    
    return render(request, 'energy_analysis/alerts.html', context)

def api_energy_stats(request):
    """API endpoint for comprehensive energy statistics"""
    try:
        df = load_energy_data()
        
        if df is None:
            return JsonResponse({
                'status': 'error',
                'message': 'Energy data not available'
            }, status=404)
        
        # Calculate costs
        df['energy_kwh'] = df['power_watt'] / 1000
        
        # Handle missing price_kwh column
        if 'price_kwh' not in df.columns:
            if 'price_kWh' in df.columns:
                df['price_kwh'] = df['price_kWh']
            elif 'Price_kWh' in df.columns:
                df['price_kwh'] = df['Price_kWh']
            elif 'price per kwh' in df.columns:
                df['price_kwh'] = df['price per kwh']
            else:
                df['price_kwh'] = 0.12  # Default price per kWh
        
        df['cost'] = df['energy_kwh'] * df['price_kwh']
        
        # Comprehensive statistics
        stats = {
            'summary': {
                'total_records': len(df),
                'total_consumption_kwh': float(df['energy_kwh'].sum()),
                'total_cost': float(df['cost'].sum()),
                'avg_consumption_w': float(df['power_watt'].mean()),
                'max_consumption_w': float(df['power_watt'].max()),
                'min_consumption_w': float(df['power_watt'].min()),
                'std_consumption_w': float(df['power_watt'].std()),
                'unique_devices': int(df['device_type'].nunique()),
                'unique_rooms': int(df['room'].nunique()),
                'unique_homes': int(df['home_id'].nunique()),
            },
            'time_analysis': {
                'start_date': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
                'peak_hour': int(df.groupby('hour_of_day')['power_watt'].mean().idxmax()),
                'off_peak_hour': int(df.groupby('hour_of_day')['power_watt'].mean().idxmin()),
            },
            'consumption_breakdown': {
                'by_device': df.groupby('device_type')['power_watt'].sum().round(2).to_dict(),
                'by_room': df.groupby('room')['power_watt'].sum().round(2).to_dict(),
                'by_home': df.groupby('home_id')['power_watt'].sum().round(2).to_dict()
            },
            'cost_breakdown': {
                'by_device': df.groupby('device_type')['cost'].sum().round(4).to_dict(),
                'by_room': df.groupby('room')['cost'].sum().round(4).to_dict(),
                'by_home': df.groupby('home_id')['cost'].sum().round(4).to_dict()
            },
            'patterns': {
                'hourly_consumption': df.groupby('hour_of_day')['power_watt'].mean().round(2).to_dict(),
                'daily_consumption': df.groupby('day_of_week')['power_watt'].mean().round(2).to_dict(),
                'hourly_cost': df.groupby('hour_of_day')['cost'].mean().round(4).to_dict()
            },
            'efficiency_metrics': {
                'user_present_avg': float(df[df['user_present'] == True]['power_watt'].mean()),
                'user_absent_avg': float(df[df['user_present'] == False]['power_watt'].mean()),
                'efficiency_ratio': float((1 - df[df['user_present'] == False]['power_watt'].mean() / df[df['user_present'] == True]['power_watt'].mean()) * 100),
                'temperature_efficiency': df.groupby(pd.cut(df['indoor_temp'], bins=5))['power_watt'].mean().to_dict()
            }
        }
        
        return JsonResponse({
            'status': 'success',
            'data': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error calculating energy statistics: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }, status=500) 
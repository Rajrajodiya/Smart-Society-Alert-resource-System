from django import forms

class PredictionForm(forms.Form):
    """Simple form for energy consumption prediction"""
    
    indoor_temp = forms.FloatField(
        label='Indoor Temperature (°C)',
        min_value=-50,
        max_value=100,
        initial=22,
        help_text='Enter indoor temperature in Celsius'
    )
    
    outdoor_temp = forms.FloatField(
        label='Outdoor Temperature (°C)',
        min_value=-50,
        max_value=100,
        initial=20,
        help_text='Enter outdoor temperature in Celsius'
    )
    
    humidity = forms.FloatField(
        label='Humidity (%)',
        min_value=0,
        max_value=100,
        initial=50,
        help_text='Enter humidity percentage'
    )
    
    light_level = forms.FloatField(
        label='Light Level (lux)',
        min_value=0,
        max_value=10000,
        initial=500,
        help_text='Enter light level in lux'
    )
    
    hour_of_day = forms.IntegerField(
        label='Hour of Day',
        min_value=0,
        max_value=23,
        initial=12,
        help_text='Enter hour of day (0-23)'
    )
    
    day_of_week = forms.IntegerField(
        label='Day of Week',
        min_value=0,
        max_value=6,
        initial=1,
        help_text='Enter day of week (0=Monday, 6=Sunday)'
    )
    
    device_type = forms.ChoiceField(
        label='Device Type',
        choices=[
            ('Appliance', 'Appliance'),
            ('HVAC', 'HVAC'),
            ('Lighting', 'Lighting'),
            ('Electronics', 'Electronics'),
            ('Kitchen', 'Kitchen'),
        ],
        initial='Appliance',
        help_text='Select device type'
    )
    
    room = forms.ChoiceField(
        label='Room',
        choices=[
            ('Living Room', 'Living Room'),
            ('Bedroom', 'Bedroom'),
            ('Kitchen', 'Kitchen'),
            ('Bathroom', 'Bathroom'),
            ('Office', 'Office'),
        ],
        initial='Living Room',
        help_text='Select room location'
    )
    
    price_kwh = forms.FloatField(
        label='Price per kWh (₹)',
        min_value=0,
        max_value=10,
        initial=0.12,
        help_text='Enter electricity price per kWh'
    )
    
    user_present = forms.BooleanField(
        label='User Present',
        required=False,
        initial=True,
        help_text='Check if user is present in the room'
    )
    
    def __init__(self, *args, device_types=None, rooms=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Update choices with dynamic data if provided
        if device_types:
            self.fields['device_type'].choices = [(dt, dt) for dt in device_types]
        if rooms:
            self.fields['room'].choices = [(r, r) for r in rooms]
        
        # Add Bootstrap classes to form fields
        for field_name, field in self.fields.items():
            field.widget.attrs.update({
                'class': 'form-control',
                'placeholder': field.help_text
            })
            # Special handling for checkbox
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs.update({
                    'class': 'form-check-input'
                })
    
    def clean_hour_of_day(self):
        hour = self.cleaned_data.get('hour_of_day')
        if hour < 0 or hour > 23:
            raise forms.ValidationError('Hour must be between 0 and 23.')
        return hour
    
    def clean_indoor_temp(self):
        temp = self.cleaned_data.get('indoor_temp')
        if temp < -50 or temp > 50:
            raise forms.ValidationError('Indoor temperature must be between -50°C and 50°C.')
        return temp
    
    def clean_outdoor_temp(self):
        temp = self.cleaned_data.get('outdoor_temp')
        if temp < -50 or temp > 50:
            raise forms.ValidationError('Outdoor temperature must be between -50°C and 50°C.')
        return temp
    
    def clean_humidity(self):
        humidity = self.cleaned_data.get('humidity')
        if humidity < 0 or humidity > 100:
            raise forms.ValidationError('Humidity must be between 0% and 100%.')
        return humidity

class ModelTrainingForm(forms.Form):
    model_type = forms.ChoiceField(
        label='Model Type',
        choices=[
            ('random_forest', 'Random Forest'),
            ('linear_regression', 'Linear Regression'),
        ],
        widget=forms.Select(attrs={'class': 'form-control'}),
        required=True
    )
    
    test_size = forms.FloatField(
        label='Test Size',
        initial=0.2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.1',
            'max': '0.5'
        }),
        required=True
    )
    
    random_state = forms.IntegerField(
        label='Random State',
        initial=42,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '0'
        }),
        required=True
    ) 
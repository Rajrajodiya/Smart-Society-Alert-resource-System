# Revert weather models to original state

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('weather', '0005_update_weather_models'),
    ]

    operations = [
        # Remove WeatherAnalysis model
        migrations.DeleteModel(
            name='WeatherAnalysis',
        ),
        
        # Remove new indexes from WeatherAlert
        migrations.RemoveIndex(
            model_name='weatheralert',
            name='weather_wea_city_a6f41a_idx',
        ),
        migrations.RemoveIndex(
            model_name='weatheralert',
            name='weather_wea_alert_t_9f7ec9_idx',
        ),
        
        # Remove new fields from WeatherAlert
        migrations.RemoveField(
            model_name='weatheralert',
            name='title',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='description',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='start_time',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='end_time',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='affected_area',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='recommendations',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='source',
        ),
        
        # Restore original fields
        migrations.AddField(
            model_name='weatheralert',
            name='message',
            field=models.TextField(default=''),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='temperature',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='humidity',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='wind_speed',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='expires_at',
            field=models.DateTimeField(default='2024-12-31 23:59:59'),
            preserve_default=False,
        ),
        
        # Revert field choices
        migrations.AlterField(
            model_name='weatheralert',
            name='alert_type',
            field=models.CharField(choices=[('EXTREME_TEMP', 'Extreme Temperature'), ('SEVERE_WEATHER', 'Severe Weather'), ('HIGH_HUMIDITY', 'High Humidity'), ('LOW_VISIBILITY', 'Low Visibility'), ('STRONG_WIND', 'Strong Wind')], max_length=20),
        ),
        migrations.AlterField(
            model_name='weatheralert',
            name='severity',
            field=models.CharField(choices=[('LOW', 'Low'), ('MEDIUM', 'Medium'), ('HIGH', 'High'), ('CRITICAL', 'Critical')], max_length=10),
        ),
    ]

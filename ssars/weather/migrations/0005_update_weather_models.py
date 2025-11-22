# Generated to update WeatherAlert model and add WeatherAnalysis

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('weather', '0004_weatheranalysis_alter_weatheralert_options_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='weatheralert',
            name='expires_at',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='humidity',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='message',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='temperature',
        ),
        migrations.RemoveField(
            model_name='weatheralert',
            name='wind_speed',
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='affected_area',
            field=models.CharField(blank=True, max_length=200),
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='description',
            field=models.TextField(default=''),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='end_time',
            field=models.DateTimeField(default='2024-01-01 00:00:00'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='recommendations',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='source',
            field=models.CharField(default='System Generated', max_length=100),
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='start_time',
            field=models.DateTimeField(default='2024-01-01 00:00:00'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='weatheralert',
            name='title',
            field=models.CharField(default='', max_length=200),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='weatheralert',
            name='alert_type',
            field=models.CharField(choices=[('storm', 'Storm Warning'), ('rain', 'Heavy Rain'), ('heat', 'Heat Wave'), ('cold', 'Cold Wave'), ('fog', 'Fog Alert'), ('wind', 'High Wind'), ('flood', 'Flood Warning'), ('snow', 'Snow Alert'), ('thunderstorm', 'Thunderstorm'), ('dust', 'Dust Storm'), ('smog', 'Smog Alert')], max_length=20),
        ),
        migrations.AlterField(
            model_name='weatheralert',
            name='severity',
            field=models.CharField(choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High'), ('critical', 'Critical')], max_length=20),
        ),
        migrations.CreateModel(
            name='WeatherAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('city', models.CharField(max_length=100)),
                ('analysis_date', models.DateField()),
                ('avg_temp', models.FloatField()),
                ('max_temp', models.FloatField()),
                ('min_temp', models.FloatField()),
                ('avg_humidity', models.FloatField()),
                ('total_precipitation', models.FloatField(default=0.0)),
                ('dominant_weather_type', models.CharField(default='', max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'ordering': ['-analysis_date'],
            },
        ),
        migrations.AddIndex(
            model_name='weatheralert',
            index=models.Index(fields=['city', 'is_active'], name='weather_wea_city_a6f41a_idx'),
        ),
        migrations.AddIndex(
            model_name='weatheralert',
            index=models.Index(fields=['alert_type', 'severity'], name='weather_wea_alert_t_9f7ec9_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='weatheranalysis',
            unique_together={('city', 'analysis_date')},
        ),
    ] 
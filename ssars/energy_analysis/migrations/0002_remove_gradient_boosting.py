# Generated manually to remove gradient_boosting choice

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('energy_analysis', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='energymodel',
            name='model_type',
            field=models.CharField(
                choices=[
                    ('random_forest', 'Random Forest'),
                    ('linear_regression', 'Linear Regression'),
                ],
                max_length=50,
            ),
        ),
    ] 
# Generated by Django 5.1.4 on 2025-01-09 22:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediccion', '0005_alter_historial_fecha_subida_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='historial',
            name='fecha_subida',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='historial',
            name='hora_subida',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]

# Generated by Django 5.1.4 on 2024-12-11 05:11

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Imagen',
            fields=[
                ('identificador', models.AutoField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('imagen', models.ImageField(upload_to='imagenes/')),
            ],
        ),
    ]

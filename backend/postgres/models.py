from django.db import models

class Imagen(models.Model):
    identificador = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=255)
    imagen = models.ImageField(upload_to='imagenes/')

    def __str__(self):
        return self.nombre
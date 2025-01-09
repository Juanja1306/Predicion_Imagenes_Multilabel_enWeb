from django.db import models
import datetime

class Historial(models.Model):
    id = models.AutoField(primary_key=True)
    prediccion = models.CharField(max_length=255)
    fecha_subida = models.CharField(max_length=50, blank=True, null=True)
    hora_subida = models.CharField(max_length=50, blank=True, null=True)
    url = models.URLField(max_length=500)

    def save(self, *args, **kwargs):
        # Establece fecha y hora actuales solo si es una nueva instancia
        if not self.id:
            ahora = datetime.datetime.now()
            self.fecha_subida = ahora.strftime('%Y-%m-%d')
            self.hora_subida = ahora.strftime('%H:%M:%S')
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.id)

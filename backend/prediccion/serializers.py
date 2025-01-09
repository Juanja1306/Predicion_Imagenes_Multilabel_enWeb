from rest_framework import serializers # type: ignore
from .models import Historial

class HistorialSerializer(serializers.ModelSerializer):

    class Meta:
        model = Historial
        fields = ['id', 'prediccion', 'fecha_subida', 'hora_subida', 'url']

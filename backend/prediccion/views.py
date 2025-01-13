import datetime
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .prediccion import predict_and_show, MLC
from .modelo import Modelo
from rest_framework.decorators import api_view # type: ignore
from rest_framework.response import Response # type: ignore
from backend.settings import bucket
from .models import Historial


from rest_framework import status # type: ignore

from .serializers import HistorialSerializer

@api_view(['POST'])
def predict_image(request):
    if request.method == 'POST':
        image_path = request.FILES['archivo']
        if image_path:
            model = MLC(num_classes=80) 
            predicted_labels = predict_and_show(image_path, model)
            
            # Crear un nuevo registro en el historial
            archivo = request.FILES['archivo']
            blob_name = archivo.name
            blob = bucket.blob(blob_name)
            
            archivo.file.seek(0)
            blob.upload_from_file(archivo.file, content_type=archivo.content_type)
            labels_string = ', '.join(predicted_labels)
        
            historial_data = {
                    'prediccion': str(labels_string),
                    'fecha_subida': "",
                    'hora_subida': "",
                    'url': blob.public_url
                }
            
            serializer = HistorialSerializer(data=historial_data)
            if serializer.is_valid():
                serializer.save()
                return JsonResponse({'predictions': predicted_labels}, status=status.HTTP_201_CREATED)
            else:
                return JsonResponse({'predictions': predicted_labels, 'errors': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        return JsonResponse({'error': 'No image provided'}, status=400)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@api_view(['POST'])
def predicion(request):
    if request.method == 'POST' and request.FILES.get('archivo'):
        # image_file = request.FILES['archivo']
        # modelo = Modelo()
        # results = modelo.predict_image(image_file)
        image_file = request.FILES['archivo']
        modelo = Modelo()
        predictions = modelo.predict_image(image_file)  
        traducido = modelo.translate_text(predictions) 
        predictions_str = ', '.join(traducido)
         
        # Crear un nuevo registro en el historial
        archivo = request.FILES['archivo']
        blob_name = archivo.name
        blob = bucket.blob(blob_name)
            
        archivo.file.seek(0)
        blob.upload_from_file(archivo.file, content_type=archivo.content_type)
        labels_string = ', '.join(predictions_str)
    
        historial_data = {
                'prediccion': str(labels_string),
                'fecha_subida': "",
                'hora_subida': "",
                'url': blob.public_url
            }

        serializer = HistorialSerializer(data=historial_data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse({'predictions': traducido}, status=status.HTTP_201_CREATED)
        else:
            return JsonResponse({'errors': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        
    else:
        return JsonResponse({'error': 'No image provided or Method not allowed'}, status=400)

@api_view(['GET'])
def historial(request):
    if request.method == 'GET':
        historial = Historial.objects.all()
        serializer = HistorialSerializer(historial, many=True)
        return Response(serializer.data)
    return Response({'error': 'Method not allowed'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
    


def home(request):
    return render(request, 'front/home.html')
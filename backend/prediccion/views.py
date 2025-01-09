from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .prediccion import predict_and_show, MLC
from rest_framework.decorators import api_view # type: ignore
from rest_framework.response import Response # type: ignore

from rest_framework import status # type: ignore

from .serializers import HistorialSerializer

@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        image_path = request.FILES.get('image')
        if image_path:
            model = MLC(num_classes=80)  # Aquí podrías cargar el modelo solo una vez al iniciar el servidor en otro escenario
            predicted_labels = predict_and_show(image_path, model)
            return JsonResponse({'predictions': predicted_labels})
        return JsonResponse({'error': 'No image provided'}, status=400)
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@api_view(['POST'])
def upload(request):
    if request.method == 'POST':
        serializer = HistorialSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    



def home(request):
    return render(request, 'front/home.html')
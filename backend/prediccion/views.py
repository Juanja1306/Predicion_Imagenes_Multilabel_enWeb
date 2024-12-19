from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .prediccion import predict_and_show, MLC


@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        image_path = request.FILES.get('image')
        if image_path:
            # Asigna el path adecuado al archivo recibido o maneja directamente el objeto imagen
            model = MLC(num_classes=80)  # Aquí podrías cargar el modelo solo una vez al iniciar el servidor en otro escenario
            predicted_labels = predict_and_show(image_path, model)
            return JsonResponse({'predictions': predicted_labels})
        return JsonResponse({'error': 'No image provided'}, status=400)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

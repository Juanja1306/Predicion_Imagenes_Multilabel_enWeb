
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from prediccion.views import predict_image   # Asegúrate de que esta importación está aquí

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', predict_image, name='predict_image'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

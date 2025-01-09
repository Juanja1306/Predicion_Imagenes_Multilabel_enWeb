
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from prediccion.views import predict_image 
from prediccion.views import historial
from prediccion import views

urlpatterns = [
    path('', views.home, name='home'),
    path('admin/', admin.site.urls),
    path('predict/', predict_image, name='predict_image'),
    path('historial/', views.historial, name='historial'),
]

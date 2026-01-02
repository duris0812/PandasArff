from django.urls import path
from . import views, views_frontend
import os

# Usar vista frontend si estamos en Render, sino usar la vista normal
if os.environ.get('DJANGO_SETTINGS_MODULE') == 'Despliegue_Final.settings_render':
    # En Render: usar vista que llama a API de Ionos
    index_view = views_frontend.index_frontend
else:
    # En Ionos: usar vista con procesamiento local
    index_view = views.index

urlpatterns = [
    path('', index_view, name='index'),
    # API endpoints (solo para Ionos)
    path('api/analyze/', views.api_analyze_dataset, name='api_analyze'),
    path('api/health/', views.api_health_check, name='api_health'),
]
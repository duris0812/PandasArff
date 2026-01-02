from django.urls import path
from . import views, views_frontend

urlpatterns = [
    # Vista original (procesamiento local) - para usar en Ionos
    path('', views.index, name='index'),
    
    # Vista frontend (llama a API) - para usar en Render
    path('frontend/', views_frontend.index_frontend, name='index_frontend'),
    
    # API endpoints
    path('api/analyze/', views.api_analyze_dataset, name='api_analyze'),
    path('api/health/', views.api_health_check, name='api_health'),
]

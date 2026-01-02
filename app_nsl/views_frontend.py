# views_frontend.py - Vista frontend que llama a la API de Ionos
from django.shortcuts import render
import requests
from io import BytesIO

# URL de tu API en Ionos
IONOS_API_URL = 'http://70.35.202.152:8000'

def index_frontend(request):
    """
    Vista frontend que renderiza el HTML y llama a la API de Ionos para procesamiento
    """
    context = {}
    
    if request.method == "POST" and request.FILES.get("dataset"):
        try:
            # Obtener el archivo subido
            uploaded_file = request.FILES["dataset"]
            
            # Preparar el archivo para enviarlo a la API
            files = {
                'dataset': (uploaded_file.name, uploaded_file.read(), uploaded_file.content_type)
            }
            
            # Llamar a la API de Ionos para procesar
            response = requests.post(
                f'{IONOS_API_URL}/api/analyze/',
                files=files,
                timeout=300  # 5 minutos timeout para datasets grandes
            )
            
            if response.status_code == 200:
                api_data = response.json()
                
                if api_data.get('success'):
                    # Extraer datos de la respuesta de la API
                    dataset_info = api_data.get('dataset_info', {})
                    statistics = api_data.get('statistics', {})
                    
                    # Preparar contexto para el template
                    context['dataset_loaded'] = True
                    context['df_shape'] = dataset_info.get('shape', [0, 0])
                    context['total_rows'] = dataset_info.get('total_rows', 0)
                    
                    # Aquí puedes agregar más procesamiento si es necesario
                    # Por ahora, la vista original hace mucho más procesamiento local
                    # Esta es una versión simplificada que muestra la información básica
                    
                    context['api_response'] = api_data
                    
                else:
                    context['error'] = api_data.get('error', 'Error desconocido en el procesamiento')
            else:
                context['error'] = f'Error al conectar con la API: {response.status_code}'
                
        except requests.exceptions.Timeout:
            context['error'] = 'Timeout: El procesamiento tomó demasiado tiempo. Intente con un dataset más pequeño.'
        except requests.exceptions.ConnectionError:
            context['error'] = 'Error de conexión: No se pudo conectar con el servidor de procesamiento en Ionos.'
        except Exception as e:
            context['error'] = f'Error procesando el archivo: {str(e)}'
    
    return render(request, 'app_nsl/index.html', context)

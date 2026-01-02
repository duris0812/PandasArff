# 🚀 Instrucciones para Despliegue Frontend/Backend Separados

## Arquitectura
- **Backend (Django API)**: Ionos VPS - IP 70.35.202.152
- **Frontend (HTML/JS)**: Render (estático)

## 📋 Paso 1: Backend en Ionos (YA CONFIGURADO)

### Lo que ya está hecho:
✅ Django REST Framework instalado
✅ CORS headers configurado
✅ ALLOWED_HOSTS actualizado
✅ Entorno virtual `.venv` creado

### Para producción en Ionos:

1. **Instalar Gunicorn** (servidor WSGI de producción):
```bash
cd /root/Despliegue_Final
.venv/bin/pip install gunicorn
```

2. **Correr con Gunicorn**:
```bash
.venv/bin/gunicorn Despliegue_Final.wsgi:application --bind 0.0.0.0:8000 --workers 3
```

3. **Crear servicio systemd** (para que corra automáticamente):
```bash
sudo nano /etc/systemd/system/django-nsl.service
```

Contenido:
```ini
[Unit]
Description=Django NSL-KDD Application
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/root/Despliegue_Final
Environment="PATH=/root/Despliegue_Final/.venv/bin"
ExecStart=/root/Despliegue_Final/.venv/bin/gunicorn \
          --workers 3 \
          --bind 0.0.0.0:8000 \
          Despliegue_Final.wsgi:application

[Install]
WantedBy=multi-user.target
```

4. **Activar el servicio**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable django-nsl
sudo systemctl start django-nsl
sudo systemctl status django-nsl
```

5. **Configurar Nginx como reverse proxy** (opcional pero recomendado):
```bash
sudo apt install nginx -y
sudo nano /etc/nginx/sites-available/django-nsl
```

Contenido:
```nginx
server {
    listen 80;
    server_name 70.35.202.152;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/django-nsl /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 📦 Paso 2: Frontend en Render

### Opción A: Extraer HTML actual y adaptarlo

1. **Extraer el template**:
El archivo actual está en `/root/Despliegue_Final/app_nsl/templates/app_nsl/index.html`

2. **Modificar para usar API**:
- Cambiar el form POST a fetch() API calls
- Actualizar URLs para apuntar a `http://70.35.202.152:8000/api/`

3. **Subir a Render**:
- Crear repositorio GitHub con el frontend
- Conectar Render a ese repo
- Tipo: Static Site

### Opción B: Crear frontend desde cero (React/Vue)

Más profesional pero requiere más tiempo.

## 🔌 Paso 3: Crear endpoints API en Django

Necesitas crear vistas API en `app_nsl/views.py`:

```python
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage

@api_view(['POST'])
def analyze_dataset(request):
    if request.FILES.get('dataset'):
        file = request.FILES['dataset']
        data_str = file.read().decode('utf-8')
        df = load_kdd_dataset(data_str)
        
        # Tu lógica de análisis aquí
        results = {
            'shape': df.shape,
            'columns': list(df.columns),
            'stats': df.describe().to_dict(),
            # ... más resultados
        }
        
        return Response(results)
    
    return Response({'error': 'No file provided'}, status=400)
```

Y agregar a `urls.py`:
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Vista actual
    path('api/analyze/', views.analyze_dataset, name='api_analyze'),
]
```

## 🌐 URLs finales:
- **Backend API**: `http://70.35.202.152:8000/api/`
- **Frontend**: `https://tu-app.onrender.com`

## ⚠️ Importante:
1. Actualiza `CORS_ALLOWED_ORIGINS` en settings.py con tu URL de Render
2. Cambia `CORS_ALLOW_ALL_ORIGINS = False` en producción
3. Genera nueva `SECRET_KEY` para producción
4. Configura `DEBUG = False` en producción

## 🔒 Firewall (si es necesario):
```bash
sudo ufw allow 8000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

## 📝 Siguiente paso inmediato:
¿Quieres que:
1. Extraiga el HTML actual y lo adapte para llamar a la API?
2. Cree los endpoints API necesarios?
3. Configure Gunicorn/Nginx para producción?

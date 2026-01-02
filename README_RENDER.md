# 🚀 Despliegue Django en Render + API en Ionos

## Arquitectura

- **Render**: Django para renderizar templates (frontend con diseño completo)
- **Ionos**: API REST para procesamiento pesado de datos

## 📋 Pasos para Desplegar en Render

### 1. Preparar el repositorio

El proyecto ya está listo. Solo necesitas subirlo a GitHub:

```bash
# Desde Windows, en la carpeta Despliegue_Final descargada
git init
git add .
git commit -m "Django frontend for Render"
git remote add origin https://github.com/duris0812/nsl-kdd-render.git
git branch -M main
git push -u origin main
```

### 2. Crear Web Service en Render

1. Ve a [Render.com](https://render.com)
2. Click en **"New +"** → **"Web Service"**
3. Conecta tu repositorio GitHub
4. **Configuración**:
   - **Name**: `nsl-kdd-frontend`
   - **Region**: Oregon (US West) - más cerca a usuarios
   - **Branch**: `main`
   - **Root Directory**: (dejar vacío)
   - **Runtime**: `Python 3`
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn Despliegue_Final.wsgi:application`

### 3. Variables de Entorno en Render

Agregar en la sección **Environment**:

```
PYTHON_VERSION=3.12.3
SECRET_KEY=tu-secret-key-super-secreta-aqui-cambiar
DEBUG=False
DJANGO_SETTINGS_MODULE=Despliegue_Final.settings_render
```

Para generar un SECRET_KEY seguro:
```python
python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 4. Deploy

Click en **"Create Web Service"** y espera 5-10 minutos.

## 🔧 Configuración Post-Despliegue

### Actualizar URL en views_frontend.py

Si cambias la IP de Ionos, edita:

```python
# app_nsl/views_frontend.py
IONOS_API_URL = 'http://TU-NUEVA-IP:8000'
```

### Verificar API en Ionos

Asegúrate de que la API esté funcionando:

```bash
curl http://70.35.202.152:8000/api/health/
```

## 📁 Archivos Clave

```
Despliegue_Final/
├── build.sh                    # Script de build para Render
├── requirements_render.txt     # Dependencias Python
├── manage.py
├── db.sqlite3
├── Despliegue_Final/
│   ├── settings.py            # Settings base
│   ├── settings_render.py     # Settings para Render ⭐
│   ├── wsgi.py
│   └── urls.py
└── app_nsl/
    ├── views.py               # Procesamiento local (para Ionos)
    ├── views_frontend.py      # Llama a API de Ionos ⭐
    ├── urls.py                # URLs originales
    ├── urls_render.py         # URLs para Render ⭐
    └── templates/
        └── app_nsl/
            └── index.html     # Template completo con diseño
```

## 🎯 Cómo Funciona

1. Usuario visita `https://tu-app.onrender.com`
2. Render renderiza el HTML con Django templates
3. Usuario sube archivo `.arff`
4. Django en Render envía archivo a API de Ionos
5. Ionos procesa (análisis, gráficos, ML)
6. Ionos devuelve resultados JSON
7. Django en Render renderiza resultados en HTML bonito
8. Usuario ve el análisis completo

## 🐛 Troubleshooting

### Error: "No se pudo conectar con el servidor"

- Verifica que la API de Ionos esté corriendo:
  ```bash
  systemctl status django-nsl
  ```

### Error: "Timeout"

- Dataset muy grande. La API de Ionos tiene timeout de 5 minutos
- Reduce el tamaño del dataset o aumenta el timeout en `views_frontend.py`

### Error: "Application failed to start"

- Revisa los logs en Render Dashboard
- Verifica que `requirements_render.txt` esté correcto
- Asegúrate de que `DJANGO_SETTINGS_MODULE` esté configurado

## ⚡ Optimizaciones

### Caché de resultados (opcional)

Puedes agregar Redis en Render para cachear resultados:

```python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL'),
    }
}
```

## 📝 URLs Finales

- **Frontend (Render)**: https://nsl-kdd-frontend.onrender.com
- **API Backend (Ionos)**: http://70.35.202.152:8000
- **API Health**: http://70.35.202.152:8000/api/health/
- **API Analyze**: http://70.35.202.152:8000/api/analyze/

## ✅ Checklist de Despliegue

- [ ] Subir código a GitHub
- [ ] Crear Web Service en Render
- [ ] Configurar variables de entorno
- [ ] Verificar que API de Ionos esté corriendo
- [ ] Probar con archivo .arff de prueba
- [ ] Verificar que se muestren los gráficos correctamente

## 🎉 ¡Listo!

Tu aplicación ahora:
- ✅ Se ve bonita en Render (todo el diseño original)
- ✅ Procesa en Ionos (siempre activo, sin límites)
- ✅ No se queda sin memoria en Render
- ✅ Mantiene TODAS las animaciones y estilos


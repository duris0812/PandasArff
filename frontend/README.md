# 🚀 Frontend NSL-KDD - Despliegue en Render

## ✅ Backend YA CONFIGURADO en Ionos

El backend Django está corriendo en producción:
- **URL**: http://70.35.202.152:8000
- **API Health**: http://70.35.202.152:8000/api/health/
- **API Analyze**: http://70.35.202.152:8000/api/analyze/
- **Servicio**: `django-nsl.service` (systemd)
- **Estado**: Activo y corriendo con Gunicorn

### Verificar backend:
```bash
sudo systemctl status django-nsl
curl http://70.35.202.152:8000/api/health/
```

### Reiniciar backend si es necesario:
```bash
sudo systemctl restart django-nsl
```

## 📦 Subir Frontend a Render

### Paso 1: Crear repositorio GitHub

1. **Crear nuevo repositorio en GitHub** (ej: `nsl-kdd-frontend`)

2. **Desde tu PC Windows**, navega a la carpeta del frontend y sube los archivos:

```bash
# Desde WSL o Git Bash en tu carpeta del proyecto
cd /ruta/al/proyecto
git init
git add .
git commit -m "Initial frontend commit"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/nsl-kdd-frontend.git
git push -u origin main
```

### Paso 2: Configurar en Render

1. Ve a [Render.com](https://render.com) y crea una cuenta (usa GitHub)

2. Click en **"New +"** → **"Static Site"**

3. Conecta tu repositorio de GitHub

4. **Configuración:**
   - **Name**: `nsl-kdd-frontend`
   - **Branch**: `main`
   - **Build Command**: (dejar vacío)
   - **Publish Directory**: `.` (punto)

5. Click en **"Create Static Site"**

6. **Espera** a que termine el despliegue (2-5 minutos)

7. Tu sitio estará disponible en: `https://nsl-kdd-frontend.onrender.com`

### Paso 3: Actualizar CORS en Django

Una vez tengas tu URL de Render, actualiza el backend:

```bash
# En el servidor Ionos
nano /root/Despliegue_Final/Despliegue_Final/settings.py
```

Busca la sección CORS y actualiza:
```python
CORS_ALLOWED_ORIGINS = [
    "https://nsl-kdd-frontend.onrender.com",  # Tu URL de Render
]

# IMPORTANTE: Cambia esto a False en producción
CORS_ALLOW_ALL_ORIGINS = False
```

Reinicia el servicio:
```bash
sudo systemctl restart django-nsl
```

## 📁 Estructura del Frontend

```
frontend/
├── index.html      # Interfaz completa (copiado del original)
├── app.js          # JavaScript para llamar a la API
└── README.md       # Este archivo
```

## 🔧 Configuración de la API

El archivo `app.js` ya está configurado para usar:
```javascript
const API_URL = 'http://70.35.202.152:8000';
```

Si cambias el puerto o dominio del backend, actualiza esta línea en `app.js`.

## 🧪 Probar Localmente

Antes de subir a Render, puedes probar localmente:

```bash
# Servidor HTTP simple con Python
cd frontend
python3 -m http.server 8080
```

Abre http://localhost:8080 en tu navegador.

## ⚠️ Importante

1. **El backend debe estar corriendo** para que el frontend funcione
2. **CORS debe estar configurado** correctamente
3. **El archivo debe ser .arff** válido

## 🐛 Troubleshooting

### Error: "API no disponible"
- Verifica que el backend esté corriendo: `systemctl status django-nsl`
- Prueba el endpoint de salud: `curl http://70.35.202.152:8000/api/health/`

### Error de CORS
- Asegúrate de que la URL de Render esté en `CORS_ALLOWED_ORIGINS`
- Reinicia el servicio Django después de cambiar settings.py

### El análisis no funciona
- Verifica el endpoint `/api/analyze/` con curl:
```bash
curl -X POST -F "dataset=@archivo.arff" http://70.35.202.152:8000/api/analyze/
```

## 📝 URLs Finales

- **Frontend (Render)**: https://tu-app.onrender.com
- **Backend API (Ionos)**: http://70.35.202.152:8000
- **API Health**: http://70.35.202.152:8000/api/health/
- **API Analyze**: http://70.35.202.152:8000/api/analyze/

## 🎉 ¡Listo!

Tu aplicación ahora está desplegada:
- **Frontend estático** en Render (gratis)
- **Backend Django API** en Ionos VPS (tu servidor)
- **Comunicación** vía API REST


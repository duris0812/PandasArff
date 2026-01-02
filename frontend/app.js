// Configuración de la API
const API_URL = 'http://70.35.202.152:8000';

// Verificar que la API esté funcionando al cargar la página
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/api/health/`);
        const data = await response.json();
        document.getElementById('apiStatus').textContent = '✅ ' + data.message;
        document.getElementById('apiStatus').style.color = 'var(--success-color)';
    } catch (error) {
        document.getElementById('apiStatus').textContent = '❌ API no disponible';
        document.getElementById('apiStatus').style.color = 'var(--danger-color)';
    }
}

// Función para analizar el dataset
async function analyzeDataset(file) {
    const formData = new FormData();
    formData.append('dataset', file);
    
    try {
        const response = await fetch(`${API_URL}/api/analyze/`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Función para mostrar los resultados
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('errorMessage');
    
    errorDiv.style.display = 'none';
    resultsDiv.style.display = 'block';
    
    // Información del dataset
    const datasetInfo = data.dataset_info;
    document.getElementById('datasetInfo').innerHTML = `
        <div class="info-grid">
            <div class="info-item">
                <strong>Filas:</strong> ${datasetInfo.total_rows.toLocaleString()}
            </div>
            <div class="info-item">
                <strong>Columnas:</strong> ${datasetInfo.total_columns}
            </div>
            <div class="info-item">
                <strong>Dimensiones:</strong> ${datasetInfo.shape[0]} × ${datasetInfo.shape[1]}
            </div>
        </div>
        <div style="margin-top: 20px;">
            <strong>Columnas del dataset:</strong>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;">
                ${datasetInfo.columns.map(col => `<span style="background: var(--gray-light); padding: 5px 12px; border-radius: 5px; font-size: 0.9em;">${col}</span>`).join('')}
            </div>
        </div>
    `;
    
    // Estadísticas
    const stats = data.statistics;
    let statsHTML = '<div style="overflow-x: auto;"><table class="stats-table"><thead><tr><th>Estadística</th>';
    
    // Encabezados de columnas
    const columns = Object.keys(stats.describe);
    columns.slice(0, 10).forEach(col => {
        statsHTML += `<th>${col}</th>`;
    });
    statsHTML += '</tr></thead><tbody>';
    
    // Filas de estadísticas
    const statRows = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
    statRows.forEach(stat => {
        statsHTML += `<tr><td><strong>${stat}</strong></td>`;
        columns.slice(0, 10).forEach(col => {
            const value = stats.describe[col][stat];
            statsHTML += `<td>${typeof value === 'number' ? value.toFixed(2) : value}</td>`;
        });
        statsHTML += '</tr>';
    });
    
    statsHTML += '</tbody></table></div>';
    document.getElementById('statistics').innerHTML = statsHTML;
    
    // Distribución de clases
    const classDistribution = data.class_distribution;
    let classHTML = '<div class="info-grid">';
    for (const [className, count] of Object.entries(classDistribution)) {
        classHTML += `
            <div class="info-item">
                <strong>Clase ${className}:</strong> ${count.toLocaleString()} registros
            </div>
        `;
    }
    classHTML += '</div>';
    document.getElementById('classDistribution').innerHTML = classHTML;
    
    // Valores faltantes
    const missingValues = stats.missing_values;
    const hasMissing = Object.values(missingValues).some(count => count > 0);
    
    if (hasMissing) {
        let missingHTML = '<div class="info-grid">';
        for (const [col, count] of Object.entries(missingValues)) {
            if (count > 0) {
                missingHTML += `
                    <div class="info-item" style="border-left: 3px solid var(--warning-color);">
                        <strong>${col}:</strong> ${count} valores faltantes
                    </div>
                `;
            }
        }
        missingHTML += '</div>';
        document.getElementById('missingValues').innerHTML = missingHTML;
    } else {
        document.getElementById('missingValues').innerHTML = '<p style="color: var(--success-color); font-weight: bold;">✅ No hay valores faltantes en el dataset</p>';
    }
}

// Función para mostrar errores
function displayError(message) {
    const errorDiv = document.getElementById('errorMessage');
    const resultsDiv = document.getElementById('results');
    
    resultsDiv.style.display = 'none';
    errorDiv.style.display = 'block';
    errorDiv.innerHTML = `<strong>⚠️ Error:</strong> ${message}`;
}

// Event listener para el formulario
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('datasetFile');
    const file = fileInput.files[0];
    
    if (!file) {
        displayError('Por favor seleccione un archivo');
        return;
    }
    
    // Mostrar estado de carga
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    
    analyzeBtn.disabled = true;
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline-block';
    
    try {
        const data = await analyzeDataset(file);
        
        if (data.success) {
            displayResults(data);
        } else {
            displayError(data.error || 'Error al procesar el dataset');
        }
    } catch (error) {
        displayError(`Error de conexión: ${error.message}. Verifique que el backend esté corriendo.`);
    } finally {
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnSpinner.style.display = 'none';
    }
});

// Verificar API al cargar la página
document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
});

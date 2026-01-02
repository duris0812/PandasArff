# views.py - Vista principal del dashboard de análisis NSL-KDD
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
import arff
import json
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import base64
from io import BytesIO
import warnings
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Imports para evaluación del modelo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

# Configuración de rendimiento
MAX_ROWS_TO_DISPLAY = 100  # Máximo de filas a mostrar en tablas HTML
SAMPLE_SIZE_FOR_STATS = 10000  # Tamaño de muestra para estadísticas rápidas

def load_kdd_dataset(data_str):
    # Comentario: Cargar dataset NSL-KDD desde string ARFF
    try:
        arff_data = arff.load(StringIO(data_str))
        attributes = [attr[0] for attr in arff_data["attributes"]]
        return pd.DataFrame(arff_data["data"], columns=attributes)
    except Exception as e:
        raise ValueError(f"Error cargando ARFF: {str(e)}")

def fig_to_base64(fig):
    # Comentario: Convertir figura matplotlib a base64 para HTML
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    # Comentario: Dividir dataset en train, validation y test sets
    strat = df[stratify] if stratify else None
    train_set, temp_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    strat = temp_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        temp_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    return train_set, val_set, test_set

def get_dataframe_sample(df, n_samples=MAX_ROWS_TO_DISPLAY):
    # Comentario: Obtener muestra representativa del DataFrame para mostrar
    if len(df) <= n_samples:
        return df
    
    # Usar muestreo estratificado si hay una columna categórica
    if 'protocol_type' in df.columns:
        try:
            # Tomar muestra proporcional de cada categoría
            sample = df.groupby('protocol_type', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(n_samples * len(x) / len(df)))))
            )
            # Si la muestra es muy pequeña, completar con random
            if len(sample) < n_samples:
                additional = df.drop(sample.index).sample(min(len(df) - len(sample), n_samples - len(sample)))
                sample = pd.concat([sample, additional])
            return sample.head(n_samples)
        except:
            return df.sample(n=min(n_samples, len(df)))
    
    return df.sample(n=min(n_samples, len(df)))

def get_large_dataframe_html(df, max_rows=MAX_ROWS_TO_DISPLAY):
    # Comentario: Generar HTML optimizado para DataFrames grandes
    if len(df) <= max_rows:
        return df.to_html(classes='table table-striped table-sm', index=False)
    
    # Obtener muestra
    sample_df = get_dataframe_sample(df, max_rows)
    
    # Crear HTML con información de filas totales
    html = sample_df.to_html(classes='table table-striped table-sm', index=False)
    
    # Añadir información sobre el total de filas
    total_info = f'<div class="table-info">'
    total_info += f'Mostrando {len(sample_df)} de {len(df)} filas totales (muestra representativa)'
    total_info += '</div>'
    
    return html + total_info

class DeleteNanRows(BaseEstimator, TransformerMixin):
    # Comentario: Transformador para eliminar filas con NaN
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.dropna()

class CustomScaler(BaseEstimator, TransformerMixin):
    # Comentario: Transformador para escalar columnas específicas
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        return X_copy

class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    # Comentario: Transformador para OneHotEncoding
    def __init__(self):
        self._oh = OneHotEncoder(sparse_output=False)
        self._columns = None
    
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])
        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh = pd.DataFrame(X_cat_oh, 
                               columns=self._columns, 
                               index=X_copy.index)
        X_copy = X_num.join(X_cat_oh)
        return X_copy

def calculate_binary_metrics(y_true, y_pred):
    # Comentario: Calcular métricas para clasificación binaria
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calcular métricas
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1-Score: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }

def index(request):
    # Comentario: Vista principal para visualización del dataset
    context = {}
    
    if request.method == "POST" and request.FILES.get("dataset"):
        try:
            # Cargar dataset desde archivo subido
            uploaded_file = request.FILES["dataset"]
            data_str = uploaded_file.read().decode('utf-8')
            df = load_kdd_dataset(data_str)
            
            # Transformar columna 'class' a numérica
            if 'class' in df.columns:
                le_class = LabelEncoder()
                df['class'] = le_class.fit_transform(df['class'].astype(str))
            
            context['dataset_loaded'] = True
            context['df_shape'] = df.shape
            context['total_rows'] = len(df)
            
            # Tema 1: Visualización del Dataset
            context['df_completo'] = get_large_dataframe_html(df)
            
            # Información básica del dataset
            buffer = StringIO()
            df.info(buf=buffer)
            info_text = buffer.getvalue()
            
            info_lines = info_text.split('\n')
            info_html = '<div class="info-content">'
            for line in info_lines:
                if line.strip():
                    info_html += f'<div>{line}</div>'
            info_html += '</div>'
            context['df_info'] = info_html
            
            # Información estadística del dataset
            context['df_describe'] = df.describe().to_html(
                classes='table table-striped table-sm'
            )
            
            # Gráfica de valores únicos de protocol_type
            if 'protocol_type' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                protocol_counts = df['protocol_type'].value_counts()
                protocol_counts.plot(kind='bar', ax=ax, color='#4a6fa5', edgecolor='black')
                ax.set_title('Distribución de protocol_type', fontsize=14, fontweight='bold')
                ax.set_xlabel('Protocol Type', fontsize=12)
                ax.set_ylabel('Frecuencia', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                context['protocol_type_histogram'] = fig_to_base64(fig)
                context['protocol_type_values'] = protocol_counts.to_dict()
            
            # Gráficas de distribución de atributos numéricos
            atributos_numericos = [
                'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
                'hot', 'num_failed_logins', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'count', 'srv_count',
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate'
            ]
            
            atributos_existentes = [col for col in atributos_numericos if col in df.columns]
            context['atributos_numericos'] = atributos_existentes
            
            if len(atributos_existentes) > 0:
                # Para datasets grandes, usar muestreo para histogramas
                plot_sample = df
                if len(df) > SAMPLE_SIZE_FOR_STATS:
                    plot_sample = df.sample(n=min(SAMPLE_SIZE_FOR_STATS, len(df)))
                
                n_atributos = len(atributos_existentes)
                n_cols = 4
                n_rows = math.ceil(n_atributos / n_cols)
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
                
                if n_rows > 1:
                    axes = axes.flatten()
                else:
                    axes = [axes] if n_cols == 1 else axes
                
                for idx, atributo in enumerate(atributos_existentes):
                    if idx < len(axes):
                        ax = axes[idx]
                        # Usar bins optimizados basados en el rango de datos
                        data = plot_sample[atributo].dropna()
                        if len(data) > 0:
                            n_bins = min(30, int(len(data) ** 0.5))
                            ax.hist(data, bins=n_bins, edgecolor='black', alpha=0.7, color='#4a6fa5')
                        ax.set_title(atributo, fontsize=11, pad=5, fontweight='bold')
                        ax.set_xlabel('Valor', fontsize=9)
                        ax.set_ylabel('Frecuencia', fontsize=9)
                        ax.tick_params(axis='both', labelsize=8)
                        ax.grid(True, alpha=0.3)
                
                for idx in range(len(atributos_existentes), len(axes)):
                    axes[idx].axis('off')
                
                plt.suptitle('Distribución de Atributos Numéricos', fontsize=16, y=1.02, fontweight='bold')
                plt.tight_layout()
                context['histogramas_cuadricula'] = fig_to_base64(fig)
            
            # Correlaciones lineales entre atributos
            if len(df) > SAMPLE_SIZE_FOR_STATS:
                corr_sample = df.sample(n=min(SAMPLE_SIZE_FOR_STATS, len(df)))
                df_numeric = corr_sample.copy()
            else:
                df_numeric = df.copy()
                
            for col in df_numeric.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
            
            numeric_df = df_numeric.select_dtypes(include=[np.number])
            correlation_matrix = numeric_df.corr()
            context['correlation_dataframe'] = correlation_matrix.to_html(
                classes='table table-striped table-sm',
                float_format='%.3f'
            )
            
            # Gráfica de matriz de correlación
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                
                # Crear una copia para modificar
                corr_display = corr.copy()
                
                # Identificar columnas especiales
                special_cols = []
                if 'num_outbound_cmds' in corr_display.columns:
                    special_cols.append('num_outbound_cmds')
                if 'is_host_login' in corr_display.columns:
                    special_cols.append('is_host_login')
                
                # Identificar columnas que deben ser 0
                zero_cols = []
                if 'logged_in' in corr_display.columns:
                    zero_cols.append('logged_in')
                if 'flag' in corr_display.columns:
                    zero_cols.append('flag')
                
                # Para logged_in y flag, establecer 0 excepto en la diagonal
                for col in zero_cols:
                    if col in corr_display.columns:
                        corr_display.loc[col, :] = 0
                        corr_display.loc[:, col] = 0
                        corr_display.loc[col, col] = 1
                
                # Para columnas especiales, establecer NaN
                for col in special_cols:
                    if col in corr_display.columns:
                        corr_display.loc[col, :] = np.nan
                        corr_display.loc[:, col] = np.nan
                
                # Asegurar valores específicos en la diagonal
                if 'urgent' in corr_display.columns:
                    corr_display.loc['urgent', 'urgent'] = 1
                
                if 'num_shells' in corr_display.columns:
                    corr_display.loc['num_shells', 'num_shells'] = 1
                
                # Quitar correlación específica
                if 'hot' in corr_display.columns and 'is_guest_login' in corr_display.columns:
                    corr_display.loc['hot', 'is_guest_login'] = 0
                    corr_display.loc['is_guest_login', 'hot'] = 0
                
                # Reemplazar NaN restantes con 0
                for i in range(len(corr_display.columns)):
                    for j in range(len(corr_display.columns)):
                        col_i = corr_display.columns[i]
                        col_j = corr_display.columns[j]
                        if col_i not in special_cols and col_j not in special_cols:
                            if pd.isna(corr_display.iloc[i, j]):
                                corr_display.iloc[i, j] = 0
                
                # Crear figura
                fig, ax = plt.subplots(figsize=(20, 18))
                
                cmap = plt.cm.viridis.copy()
                cmap.set_bad(color='white')
                
                mat = ax.matshow(corr_display, cmap=cmap, vmin=-1, vmax=1)
                
                # Configurar etiquetas
                ax.set_xticks(range(len(corr_display.columns)))
                ax.set_yticks(range(len(corr_display.columns)))
                ax.set_xticklabels(corr_display.columns, rotation=90, ha='left', fontsize=9)
                ax.set_yticklabels(corr_display.columns, fontsize=9)
                
                ax.set_title('Matriz de Correlación', fontsize=22, pad=20, fontweight='bold')
                
                # Añadir barra de color
                cbar = plt.colorbar(mat, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Correlación', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                context['correlation_matrix_plot'] = fig_to_base64(fig)
            
            # Gráfica de scatter matrix
            scatter_variables = ["same_srv_rate", "dst_host_srv_count", "class", "dst_host_same_srv_rate"]
            scatter_variables_existentes = [var for var in scatter_variables if var in df.columns]
            context['scatter_variables'] = scatter_variables_existentes
            
            if len(scatter_variables_existentes) >= 2:
                scatter_df = df[scatter_variables_existentes]
                
                # Crear scatter matrix con todos los puntos
                fig = plt.figure(figsize=(16, 16))
                scatter_matrix(scatter_df, ax=fig.gca())
                
                plt.suptitle('Scatter Matrix - Todos los puntos del dataset', fontsize=20, y=1.02, fontweight='bold')
                plt.tight_layout()
                context['scatter_matrix_plot'] = fig_to_base64(fig)
            
            # Tema 2: División del Dataset
            if 'protocol_type' in df.columns:
                # Dividir el dataset usando stratified sampling
                train_set, val_set, test_set = train_val_test_split(
                    df, stratify="protocol_type", rstate=42
                )
                
                # Guardar los sets en contexto para uso posterior
                context['train_set'] = train_set
                context['val_set'] = val_set
                context['test_set'] = test_set
                
                # Estadísticas de la división
                context['division_stats'] = {
                    'total_rows': len(df),
                    'train_rows': len(train_set),
                    'val_rows': len(val_set),
                    'test_rows': len(test_set),
                    'train_percent': round(len(train_set) / len(df) * 100, 1),
                    'val_percent': round(len(val_set) / len(df) * 100, 1),
                    'test_percent': round(len(test_set) / len(df) * 100, 1)
                }
                
                # Gráficas para el grid 2x2
                # Gráfica 1: Distribución en dataset completo
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                df_counts = df['protocol_type'].value_counts()
                bars1 = ax1.bar(df_counts.index, df_counts.values, color='#2c3e50', edgecolor='black')
                ax1.set_title('Dataset Completo', fontsize=12, pad=10, fontweight='bold')
                ax1.set_xlabel('Protocol Type', fontsize=10)
                ax1.set_ylabel('Frecuencia', fontsize=10)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                context['division_plot_full'] = fig_to_base64(fig1)
                
                # Gráfica 2: Distribución en train set
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                train_counts = train_set['protocol_type'].value_counts()
                bars2 = ax2.bar(train_counts.index, train_counts.values, color='#27ae60', edgecolor='black')
                ax2.set_title('Train Set (60%)', fontsize=12, pad=10, fontweight='bold')
                ax2.set_xlabel('Protocol Type', fontsize=10)
                ax2.set_ylabel('Frecuencia', fontsize=10)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                context['division_plot_train'] = fig_to_base64(fig2)
                
                # Gráfica 3: Distribución en validation set
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                val_counts = val_set['protocol_type'].value_counts()
                bars3 = ax3.bar(val_counts.index, val_counts.values, color='#f39c12', edgecolor='black')
                ax3.set_title('Validation Set (20%)', fontsize=12, pad=10, fontweight='bold')
                ax3.set_xlabel('Protocol Type', fontsize=10)
                ax3.set_ylabel('Frecuencia', fontsize=10)
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
                
                for bar in bars3:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                context['division_plot_val'] = fig_to_base64(fig3)
                
                # Gráfica 4: Distribución en test set
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                test_counts = test_set['protocol_type'].value_counts()
                bars4 = ax4.bar(test_counts.index, test_counts.values, color='#e74c3c', edgecolor='black')
                ax4.set_title('Test Set (20%)', fontsize=12, pad=10, fontweight='bold')
                ax4.set_xlabel('Protocol Type', fontsize=10)
                ax4.set_ylabel('Frecuencia', fontsize=10)
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
                
                for bar in bars4:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                context['division_plot_test'] = fig_to_base64(fig4)
                
                # Tema 3: Preparación del Dataset
                # Separar características (X) y etiquetas (y)
                X_train_set = train_set.drop("class", axis=1)
                y_train_set = train_set["class"].copy()
                
                # Información de X_train después de separar
                buffer_x = StringIO()
                X_train_set.info(buf=buffer_x)
                x_info_text = buffer_x.getvalue()
                
                x_info_lines = x_info_text.split('\n')
                x_info_html = '<div class="info-content">'
                for line in x_info_lines:
                    if line.strip():
                        x_info_html += f'<div>{line}</div>'
                x_info_html += '</div>'
                context['x_train_info'] = x_info_html
                
                # Crear valores nulos intencionalmente
                X_train_with_nan = X_train_set.copy()
                if 'src_bytes' in X_train_with_nan.columns and 'dst_bytes' in X_train_with_nan.columns:
                    mask_src = (X_train_with_nan["src_bytes"] > 400) & (X_train_with_nan["src_bytes"] < 800)
                    mask_dst = (X_train_with_nan["dst_bytes"] > 500) & (X_train_with_nan["dst_bytes"] < 2000)
                    
                    X_train_with_nan.loc[mask_src, "src_bytes"] = np.nan
                    X_train_with_nan.loc[mask_dst, "dst_bytes"] = np.nan
                
                # Verificar columnas con valores nulos
                columns_with_nan = X_train_with_nan.columns[X_train_with_nan.isna().any()].tolist()
                context['columns_with_nan'] = columns_with_nan
                
                # Mostrar muestra de filas con valores nulos
                filas_con_nan = X_train_with_nan[X_train_with_nan.isnull().any(axis=1)]
                context['filas_con_nan'] = get_large_dataframe_html(filas_con_nan)
                
                # Manejo de valores nulos - 3 opciones
                X_train_sin_filas_nan = X_train_with_nan.copy()
                X_train_sin_filas_nan.dropna(subset=["src_bytes", "dst_bytes"], inplace=True)
                context['filas_eliminadas_count'] = len(X_train_with_nan) - len(X_train_sin_filas_nan)
                
                X_train_sin_columnas_nan = X_train_with_nan.copy()
                X_train_sin_columnas_nan.drop(["src_bytes", "dst_bytes"], axis=1, inplace=True)
                context['columnas_restantes'] = len(X_train_sin_columnas_nan.columns)
                
                # Rellenar con mediana
                X_train_imputed = X_train_with_nan.copy()
                numeric_cols = X_train_imputed.select_dtypes(include=[np.number]).columns
                
                imputer = SimpleImputer(strategy="median")
                X_train_imputed_num = imputer.fit_transform(X_train_imputed[numeric_cols])
                X_train_imputed[numeric_cols] = X_train_imputed_num
                
                # Comparación antes/después del imputing
                comparison_data = []
                nan_indices = filas_con_nan.index[:MAX_ROWS_TO_DISPLAY]
                
                for idx in nan_indices:
                    if idx < len(X_train_with_nan):
                        row = {
                            'idx': idx,
                            'src_bytes_original': X_train_with_nan.loc[idx, 'src_bytes'],
                            'src_bytes_imputed': X_train_imputed.loc[idx, 'src_bytes'],
                            'dst_bytes_original': X_train_with_nan.loc[idx, 'dst_bytes'],
                            'dst_bytes_imputed': X_train_imputed.loc[idx, 'dst_bytes']
                        }
                        comparison_data.append(row)
                
                if comparison_data:
                    context['comparison_imputing'] = pd.DataFrame(comparison_data).to_html(
                        classes='table table-striped table-sm',
                        index=False
                    )
                
                # Transformación de categóricos a numéricos
                if 'protocol_type' in X_train_set.columns:
                    # Método 1: Pandas factorize
                    protocol_factorized, protocol_categories = X_train_set['protocol_type'].factorize()
                    
                    # Mostrar muestra de valores factorizados
                    sample_size = min(MAX_ROWS_TO_DISPLAY, len(X_train_set))
                    sample_indices = X_train_set.index[:sample_size]
                    
                    factorized_examples = []
                    for idx in sample_indices:
                        factorized_examples.append({
                            'original': X_train_set.loc[idx, 'protocol_type'],
                            'factorized': protocol_factorized[X_train_set.index.get_loc(idx)]
                        })
                    
                    context['factorized_examples'] = pd.DataFrame(factorized_examples).to_html(
                        classes='table table-striped table-sm',
                        index=False
                    )
                    context['factorized_categories'] = list(protocol_categories)
                    
                    # Método 2: OrdinalEncoder
                    protocol_df = X_train_set[["protocol_type"]]
                    ordinal_encoder = OrdinalEncoder()
                    protocol_ordinal = ordinal_encoder.fit_transform(protocol_df)
                    
                    # Método 3: OneHotEncoder
                    oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    protocol_oh = oh_encoder.fit_transform(protocol_df)
                    
                    # Mostrar muestra de OneHotEncoder
                    sample_oh = protocol_oh[:sample_size]
                    oh_df = pd.DataFrame(sample_oh, 
                                       columns=[f'protocol_{i}' for i in range(protocol_oh.shape[1])])
                    context['onehot_structure'] = oh_df.to_html(
                        classes='table table-striped table-sm'
                    )
                    
                    # Comparación de métodos de encoding
                    comparison_methods = []
                    for i, idx in enumerate(sample_indices):
                        comparison_methods.append({
                            'original': X_train_set.loc[idx, 'protocol_type'],
                            'factorize': protocol_factorized[X_train_set.index.get_loc(idx)],
                            'ordinal': protocol_ordinal[X_train_set.index.get_loc(idx)][0],
                            'onehot': str(protocol_oh[X_train_set.index.get_loc(idx)])
                        })
                    
                    context['encoding_comparison'] = pd.DataFrame(comparison_methods).to_html(
                        classes='table table-striped table-sm',
                        index=False
                    )
                    
                    # Get Dummies (método Pandas)
                    protocol_dummies = pd.get_dummies(X_train_set['protocol_type'].iloc[:sample_size])
                    context['get_dummies_sample'] = protocol_dummies.to_html(
                        classes='table table-striped table-sm'
                    )
                
                # Escalado del dataset
                if 'src_bytes' in X_train_set.columns and 'dst_bytes' in X_train_set.columns:
                    # Seleccionar columnas para escalar
                    scale_cols = ['src_bytes', 'dst_bytes']
                    
                    # Aplicar RobustScaler
                    robust_scaler = RobustScaler()
                    X_train_scaled_array = robust_scaler.fit_transform(X_train_set[scale_cols])
                    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=scale_cols)
                    
                    # Mostrar muestra de comparación antes/después del escalado
                    sample_indices = X_train_set.index[:min(MAX_ROWS_TO_DISPLAY, len(X_train_set))]
                    scaling_comparison = []
                    
                    for idx in sample_indices:
                        i = X_train_set.index.get_loc(idx)
                        scaling_comparison.append({
                            'idx': idx,
                            'src_bytes_original': X_train_set.loc[idx, 'src_bytes'],
                            'src_bytes_scaled': X_train_scaled.iloc[i]['src_bytes'],
                            'dst_bytes_original': X_train_set.loc[idx, 'dst_bytes'],
                            'dst_bytes_scaled': X_train_scaled.iloc[i]['dst_bytes']
                        })
                    
                    context['scaling_comparison'] = pd.DataFrame(scaling_comparison).to_html(
                        classes='table table-striped table-sm',
                        index=False
                    )
                    
                    # Estadísticas del escalado
                    context['scaling_stats'] = {
                        'src_bytes_mean_original': round(X_train_set['src_bytes'].mean(), 2),
                        'src_bytes_mean_scaled': round(X_train_scaled['src_bytes'].mean(), 2),
                        'src_bytes_std_original': round(X_train_set['src_bytes'].std(), 2),
                        'src_bytes_std_scaled': round(X_train_scaled['src_bytes'].std(), 2),
                        'dst_bytes_mean_original': round(X_train_set['dst_bytes'].mean(), 2),
                        'dst_bytes_mean_scaled': round(X_train_scaled['dst_bytes'].mean(), 2),
                        'dst_bytes_std_original': round(X_train_set['dst_bytes'].std(), 2),
                        'dst_bytes_std_scaled': round(X_train_scaled['dst_bytes'].std(), 2)
                    }
                
                # Tema 4: Transformadores y Pipelines
                # Transformador DeleteNanRows
                delete_nan = DeleteNanRows()
                X_train_no_nan = delete_nan.fit_transform(X_train_with_nan)
                context['delete_nan_shape'] = X_train_no_nan.shape
                context['delete_nan_sample'] = get_large_dataframe_html(X_train_no_nan)
                
                # Transformador CustomScaler
                custom_scaler = CustomScaler(["src_bytes", "dst_bytes"])
                X_train_scaled_custom = custom_scaler.fit_transform(
                    X_train_set[['src_bytes', 'dst_bytes']].sample(n=min(MAX_ROWS_TO_DISPLAY, len(X_train_set)))
                )
                context['custom_scaler_sample'] = X_train_scaled_custom.to_html(
                    classes='table table-striped table-sm'
                )
                
                # Transformador CustomOneHotEncoding
                if 'protocol_type' in X_train_set.columns:
                    custom_onehot = CustomOneHotEncoding()
                    X_train_sample = X_train_set[['protocol_type']].sample(n=min(MAX_ROWS_TO_DISPLAY, len(X_train_set)))
                    X_train_onehot = custom_onehot.fit_transform(X_train_sample)
                    context['custom_onehot_shape'] = X_train_onehot.shape
                    context['custom_onehot_sample'] = X_train_onehot.to_html(
                        classes='table table-striped table-sm'
                    )
                
                # Pipeline para atributos numéricos
                num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy="median")),
                    ('rbst_scaler', RobustScaler()),
                ])
                
                X_train_num = X_train_set.select_dtypes(exclude=['object'])
                X_train_pipeline = num_pipeline.fit_transform(X_train_num)
                X_train_pipeline_df = pd.DataFrame(X_train_pipeline, 
                                                  columns=X_train_num.columns, 
                                                  index=X_train_num.index)
                
                # Mostrar muestra del pipeline
                context['num_pipeline_sample'] = get_large_dataframe_html(
                    X_train_pipeline_df.sample(n=min(MAX_ROWS_TO_DISPLAY, len(X_train_pipeline_df)))
                )
                context['num_pipeline_shape'] = X_train_pipeline_df.shape
                
                # Comparación antes/después del pipeline
                sample_indices = X_train_num.index[:min(MAX_ROWS_TO_DISPLAY, len(X_train_num))]
                comparison_pipeline = []
                
                for idx in sample_indices:
                    i = X_train_num.index.get_loc(idx)
                    row = {
                        'idx': idx,
                        'src_bytes_original': X_train_num.loc[idx, 'src_bytes'],
                        'src_bytes_pipeline': X_train_pipeline_df.iloc[i]['src_bytes'],
                        'dst_bytes_original': X_train_num.loc[idx, 'dst_bytes'],
                        'dst_bytes_pipeline': X_train_pipeline_df.iloc[i]['dst_bytes']
                    }
                    comparison_pipeline.append(row)
                
                context['pipeline_comparison'] = pd.DataFrame(comparison_pipeline).to_html(
                    classes='table table-striped table-sm',
                    index=False
                )
                
                # ColumnTransformer completo
                num_attribs = list(X_train_set.select_dtypes(exclude=['object']))
                cat_attribs = list(X_train_set.select_dtypes(include=['object']))
                
                full_pipeline = ColumnTransformer([
                    ("num", num_pipeline, num_attribs),
                    ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_attribs)
                ])
                
                X_train_full = full_pipeline.fit_transform(X_train_set)
                # Obtener nombres de columnas después del OneHotEncoding
                cat_encoder = full_pipeline.named_transformers_['cat']
                cat_onehot_columns = list(cat_encoder.get_feature_names_out(cat_attribs))
                all_columns = num_attribs + cat_onehot_columns
                
                X_train_full_df = pd.DataFrame(X_train_full, 
                                              columns=all_columns,
                                              index=X_train_set.index)
                
                # Mostrar muestra del ColumnTransformer
                context['full_pipeline_sample'] = get_large_dataframe_html(
                    X_train_full_df.sample(n=min(MAX_ROWS_TO_DISPLAY, len(X_train_full_df)))
                )
                context['full_pipeline_shape'] = X_train_full_df.shape
                context['full_pipeline_columns'] = all_columns[:50]
                
                # Tema 5: Evaluación del Modelo
                # Preparar datos para entrenamiento
                X_train_model = train_set.drop("class", axis=1)
                y_train_model = train_set["class"].copy()
                
                X_val_model = val_set.drop("class", axis=1)
                y_val_model = val_set["class"].copy()
                
                # Preparar datos usando el pipeline
                X_train_prep = full_pipeline.fit_transform(X_train_model)
                X_val_prep = full_pipeline.transform(X_val_model)
                
                # Entrenar modelo de Regresión Logística
                log_reg = LogisticRegression(max_iter=500, random_state=42)
                log_reg.fit(X_train_prep, y_train_model)
                
                # Hacer predicciones en conjunto de validación
                y_pred = log_reg.predict(X_val_prep)
                
                # Verificar si es binario o multiclase
                unique_classes = np.unique(y_val_model)
                num_classes = len(unique_classes)
                context['num_classes'] = num_classes
                
                # Calcular métricas correctamente
                if num_classes == 2:
                    # Para clasificación binaria
                    metrics = calculate_binary_metrics(y_val_model, y_pred)
                    context['confusion_matrix'] = metrics['confusion_matrix']
                    
                    context['model_metrics'] = {
                        'accuracy': round(metrics['accuracy'] * 100, 2),
                        'precision': round(metrics['precision'] * 100, 2),
                        'recall': round(metrics['recall'] * 100, 2),
                        'f1': round(metrics['f1'] * 100, 2)
                    }
                else:
                    # Para clasificación multiclase
                    accuracy = accuracy_score(y_val_model, y_pred)
                    precision = precision_score(y_val_model, y_pred, average='macro', zero_division=0)
                    recall = recall_score(y_val_model, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_val_model, y_pred, average='macro', zero_division=0)
                    
                    context['model_metrics'] = {
                        'accuracy': round(accuracy * 100, 2),
                        'precision': round(precision * 100, 2),
                        'recall': round(recall * 100, 2),
                        'f1': round(f1 * 100, 2)
                    }
                    
                    # Matriz de confusión para multiclase
                    cm = confusion_matrix(y_val_model, y_pred)
                    context['confusion_matrix'] = cm.tolist()
                
                # Comparación y_val vs y_pred
                sample_size = min(MAX_ROWS_TO_DISPLAY, len(y_val_model))
                comparison_predictions = []
                
                for i in range(sample_size):
                    comparison_predictions.append({
                        'indice': i,
                        'y_val_real': y_val_model.iloc[i],
                        'y_pred': y_pred[i],
                        'correcto': '✅' if y_val_model.iloc[i] == y_pred[i] else '❌'
                    })
                
                context['comparison_predictions'] = pd.DataFrame(comparison_predictions).to_html(
                    classes='table table-striped table-sm',
                    index=False
                )
                
                # Estadísticas de predicciones
                context['prediction_stats'] = {
                    'total_predictions': len(y_pred),
                    'correct_predictions': int(sum(y_val_model.values == y_pred)),
                    'incorrect_predictions': int(sum(y_val_model.values != y_pred)),
                }
                
                # Información sobre el cálculo
                if num_classes > 2:
                    context['metrics_info'] = f"Dataset multiclase ({num_classes} clases). Se usó 'macro' average para Precision, Recall y F1."
            
            context['dataset_loaded'] = True
            context['df_shape'] = df.shape
            context['optimized'] = True
            
        except Exception as e:
            context['error'] = f"Error procesando el archivo: {str(e)}"
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
    
    return render(request, 'app_nsl/index.html', context)


# ============= API ENDPOINTS =============

@api_view(['POST'])
def api_analyze_dataset(request):
    """
    Endpoint API para analizar dataset NSL-KDD
    Recibe archivo ARFF y retorna análisis en JSON
    """
    try:
        if not request.FILES.get('dataset'):
            return Response(
                {'error': 'No se proporcionó archivo dataset'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Cargar dataset
        uploaded_file = request.FILES['dataset']
        data_str = uploaded_file.read().decode('utf-8')
        df = load_kdd_dataset(data_str)
        
        # Transformar columna 'class' a numérica
        if 'class' in df.columns:
            le_class = LabelEncoder()
            df['class'] = le_class.fit_transform(df['class'].astype(str))
        
        # Análisis básico
        result = {
            'success': True,
            'dataset_info': {
                'shape': list(df.shape),
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
            },
            'statistics': {
                'describe': df.describe().to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict(),
            },
            'class_distribution': df['class'].value_counts().to_dict() if 'class' in df.columns else {},
        }
        
        return Response(result, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {'error': f'Error procesando archivo: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def api_health_check(request):
    """
    Endpoint para verificar que la API está funcionando
    """
    return Response({
        'status': 'OK',
        'message': 'API NSL-KDD funcionando correctamente',
        'version': '1.0'
    })
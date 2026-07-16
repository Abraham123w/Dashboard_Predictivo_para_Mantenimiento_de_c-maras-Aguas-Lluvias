import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

# 1. Cargar datos
print("--- Iniciando Modelo Predictivo ---")
print("Cargando datos...")
df = pd.read_csv('1.data.csv', sep=';', encoding='latin-1')

# --- CORRECCIÓN DE DECIMALES ---
# Las coordenadas vienen con coma (ej: -72,56) y Python necesita punto (ej: -72.56)
# Convertimos las columnas de coordenadas de texto a números flotantes
cols_coords = ['Coordenada X', 'Coordenada Y']
for col in cols_coords:
    # Verificamos si la columna es de tipo objeto (texto) para reemplazar
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

df['Fecha_Formato'] = pd.to_datetime(df['Fecha'], dayfirst=True)

# Limpieza básica: Ordenar y eliminar duplicados exactos (mismo día y sector)
df = df.sort_values(by=['Ubicación_Microsector', 'Fecha_Formato'])
df['Fecha_Solo_Dia'] = df['Fecha_Formato'].dt.date
df = df.drop_duplicates(subset=['Ubicación_Microsector', 'Fecha_Solo_Dia'])

# 2. Ingeniería de Características (Feature Engineering)
# Preparamos las variables categóricas ANTES de dividir datos para que el codificador conozca todas las etiquetas
print("Procesando variables categóricas y coordenadas...")
le_macro = LabelEncoder()
le_micro = LabelEncoder()

df['Macrosector_Code'] = le_macro.fit_transform(df['Macrosector'])
df['Microsector_Code'] = le_micro.fit_transform(df['Ubicación_Microsector'])

# 3. Creación del Dataset de Entrenamiento
# Objetivo: Predecir el 'Intervalo' de días hasta la PRÓXIMA limpieza
# Calculamos la diferencia con la fila siguiente (del mismo sector)
df['Proxima_Fecha'] = df.groupby('Ubicación_Microsector')['Fecha_Formato'].shift(-1)
df['Dias_Hasta_Proximo_Evento'] = (df['Proxima_Fecha'] - df['Fecha_Formato']).dt.days

# Variables predictoras temporales (para capturar estacionalidad)
df['Mes'] = df['Fecha_Formato'].dt.month
df['Dia_Ano'] = df['Fecha_Formato'].dt.dayofyear

# Filtramos para crear el set de entrenamiento:
# Eliminamos las filas donde no sabemos cuándo fue la próxima limpieza (las últimas de cada historial o datos únicos)
df_train = df.dropna(subset=['Dias_Hasta_Proximo_Evento'])
# Filtramos errores de datos (días negativos o cero)
df_train = df_train[df_train['Dias_Hasta_Proximo_Evento'] > 0]

print(f"Registros útiles para entrenamiento (historial de intervalos): {len(df_train)}")

# Definimos X (Predictoras) e y (Objetivo)
features = ['Macrosector_Code', 'Microsector_Code', 'Coordenada X', 'Coordenada Y', 'Mes', 'Dia_Ano']
target = 'Dias_Hasta_Proximo_Evento'

X = df_train[features]
y = df_train[target]

# 4. Entrenamiento del Modelo
print("Entrenando Random Forest Regressor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuración del modelo: 100 árboles de decisión
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluación rápida
y_pred_test = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Precisión del modelo (Error Promedio RMSE): +/- {rmse:.1f} días")

# 5. Predicción para el Estado Actual (Proyección Inteligente)
print("Generando predicciones para el estado actual de los sectores...")

# Para predecir el futuro, tomamos el ÚLTIMO registro conocido de cada sector
# Usaremos sus características para preguntar al modelo: "¿Cuánto falta para la siguiente?"
df_ultimos = df.sort_values('Fecha_Formato').groupby('Ubicación_Microsector').tail(1).copy()

X_actual = df_ultimos[features]

# El modelo predice el intervalo ideal basado en las características de la última limpieza
intervalos_predichos = rf_model.predict(X_actual)

df_ultimos['Intervalo_Estimado_Modelo'] = intervalos_predichos

# Calculamos la fecha sugerida
df_ultimos['Fecha_Proxima_Predicha'] = df_ultimos['Fecha_Formato'] + pd.to_timedelta(df_ultimos['Intervalo_Estimado_Modelo'], unit='D')

# Comparamos con HOY
hoy = pd.Timestamp.now().normalize()
df_ultimos['Fecha_Hoy'] = hoy
df_ultimos['Dias_Restantes_Prediccion'] = (df_ultimos['Fecha_Proxima_Predicha'] - hoy).dt.days

# Clasificación de Prioridad
def clasificar_estado(dias):
    if dias < 0:
        return '🔴 Atrasado (Prioridad Alta)'
    elif dias <= 7:
        return '🟡 Próximo (Semana entrante)'
    else:
        return '🟢 A tiempo'

df_ultimos['Estado_IA'] = df_ultimos['Dias_Restantes_Prediccion'].apply(clasificar_estado)

# 6. Exportar Resultados
# Agregamos las Coordenadas X e Y a la lista de exportación
cols_exportar = [
    'Ubicación_Microsector', 
    'Coordenada X',
    'Coordenada Y',
    'Fecha_Formato', # Última limpieza real
    'Intervalo_Estimado_Modelo', # Lo que la IA cree que debe durar
    'Fecha_Proxima_Predicha', 
    'Dias_Restantes_Prediccion', 
    'Estado_IA'
]

resultado_final = df_ultimos[cols_exportar].sort_values('Dias_Restantes_Prediccion')

# Formateo para Excel
resultado_final['Fecha_Formato'] = resultado_final['Fecha_Formato'].dt.strftime('%d-%m-%Y')
resultado_final['Fecha_Proxima_Predicha'] = resultado_final['Fecha_Proxima_Predicha'].dt.strftime('%d-%m-%Y')
resultado_final['Intervalo_Estimado_Modelo'] = resultado_final['Intervalo_Estimado_Modelo'].round(1)

print("\n--- Top 10 Prioridades según Inteligencia Artificial ---")
print(resultado_final.head(10).to_string(index=False))

nombre_archivo = 'prediccion_inteligente_rf.csv'
resultado_final.to_csv(nombre_archivo, index=False, sep=';', encoding='utf-8-sig')
print(f"\nArchivo generado exitosamente: '{nombre_archivo}'")
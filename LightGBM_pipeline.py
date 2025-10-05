import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # Para guardar y cargar el pipeline

# --- 1. Creación de Datos Sintéticos para Churn Bancario ---
# Para que los resultados sean reproducibles
np.random.seed(42)

print("Generando datos sintéticos de clientes...")
n_clientes = 5000
data = {
    'edad': np.random.randint(22, 70, n_clientes),
    'saldo_cuenta': np.random.uniform(0, 250000, n_clientes),
    'antiguedad_cliente_meses': np.random.randint(0, 120, n_clientes),
    'num_productos': np.random.randint(1, 5, n_clientes),
    'tiene_tarjeta_credito': np.random.randint(0, 2, n_clientes),
    'es_miembro_activo': np.random.randint(0, 2, n_clientes),
    'salario_estimado': np.random.uniform(20000, 200000, n_clientes)
}
df = pd.DataFrame(data)

# Generamos la variable objetivo 'churn' (1 si se va, 0 si se queda)
# Hacemos que el churn sea más probable para clientes con bajo saldo, pocos productos o inactivos.
probabilidad_churn = (df['saldo_cuenta'] < 10000) * 0.2 + \
                     (df['num_productos'] == 1) * 0.15 + \
                     (df['es_miembro_activo'] == 0) * 0.1 + \
                     (df['antiguedad_cliente_meses'] < 12) * 0.05
churn = (probabilidad_churn > np.random.rand(n_clientes) * 0.4).astype(int)
df['churn'] = churn

print(f"\nDatos generados. Total de clientes: {len(df)}")
print(f"Distribución de Churn:\n{df['churn'].value_counts(normalize=True)}\n")
print("Primeras 5 filas de los datos:")
print(df.head())


# --- 2. Preparación de los Datos ---
# Separamos las características (X) de la variable objetivo (y)
X = df.drop('churn', axis=1)
y = df['churn']

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nDatos divididos en {len(X_train)} para entrenamiento y {len(X_test)} para prueba.")


# --- 3. Creación y Entrenamiento del Pipeline ---
# El pipeline contendrá dos pasos:
# 1. 'scaler': Estandariza las características numéricas (media 0, desviación estándar 1).
#    LightGBM no lo requiere estrictamente, pero es una buena práctica para muchos otros modelos
#    y no afecta negativamente a los modelos basados en árboles.
# 2. 'classifier': El modelo de clasificación LightGBM.

print("\nCreando y entrenando el pipeline...")

pipeline_lgbm = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])

# Entrenamos el pipeline completo con los datos de entrenamiento
pipeline_lgbm.fit(X_train, y_train)

print("¡Pipeline entrenado con éxito!")


# --- 4. Evaluación del Pipeline ---
# Usamos el pipeline entrenado para hacer predicciones en el conjunto de prueba
print("\nEvaluando el rendimiento del pipeline en el conjunto de prueba...")
y_pred = pipeline_lgbm.predict(X_test)

# Mostramos las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Precisión Global): {accuracy:.4f}")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))


# --- 5. Productivización: Guardar y Cargar el Pipeline ---
# Ahora, este único objeto 'pipeline_lgbm' puede ser guardado y desplegado.
print("\nGuardando el pipeline en un archivo 'pipeline_churn.joblib'...")
joblib.dump(pipeline_lgbm, 'pipeline_churn.joblib')

# Simulamos un entorno de producción donde cargamos el pipeline
print("Cargando el pipeline desde el archivo...")
pipeline_cargado = joblib.load('pipeline_churn.joblib')

# Creamos un nuevo cliente (como un diccionario o DataFrame) para predecir
nuevo_cliente = pd.DataFrame([{
    'edad': 45,
    'saldo_cuenta': 500,       # Saldo bajo
    'antiguedad_cliente_meses': 6, # Poca antiguedad
    'num_productos': 1,        # Solo un producto
    'tiene_tarjeta_credito': 1,
    'es_miembro_activo': 0,    # Inactivo
    'salario_estimado': 75000
}])

print("\nPrediciendo para un nuevo cliente con alto riesgo de churn:")
print(nuevo_cliente)

# Usamos el pipeline cargado para predecir.
# Nota: No necesitamos escalar los datos manualmente, el pipeline lo hace por nosotros.
prediccion_nuevo_cliente = pipeline_cargado.predict(nuevo_cliente)
probabilidades_nuevo_cliente = pipeline_cargado.predict_proba(nuevo_cliente)

print(f"\nPredicción de Churn (1=Sí, 0=No): {prediccion_nuevo_cliente[0]}")
print(f"Probabilidades (Clase 0, Clase 1): {probabilidades_nuevo_cliente[0]}")


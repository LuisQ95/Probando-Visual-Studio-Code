# app.py - Un servidor web simple con Flask
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# 1. Inicializar la aplicación Flask
app = Flask(__name__)

# 2. Cargar el pipeline de machine learning que ya entrenaste
try:
    pipeline = joblib.load('pipeline_churn.joblib')
    print("Pipeline cargado exitosamente.")
except FileNotFoundError:
    pipeline = None
    print("Error: No se encontró el archivo 'pipeline_churn.joblib'. Asegúrate de ejecutar primero el script de entrenamiento.")

# Ruta de bienvenida para verificar que el servidor está funcionando
@app.route('/', methods=['GET'])
def home():
    return "<h1>Servidor de predicción de Churn</h1><p>El servidor está funcionando. Usa el endpoint /predict para hacer predicciones.</p>"

# 3. Definir un "endpoint" para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Modelo no disponible'}), 500

    # Obtener los datos JSON enviados en la petición
    json_data = request.get_json()
    
    # Convertir los datos a un DataFrame de pandas
    df_cliente = pd.DataFrame(json_data, index=[0])
    
    # Usar el pipeline para predecir
    prediccion = pipeline.predict(df_cliente)
    probabilidades = pipeline.predict_proba(df_cliente)
    
    # Devolver el resultado como JSON
    resultado = {
        'prediccion_churn': int(prediccion[0]),
        'probabilidad_no_churn': round(probabilidades[0][0], 4),
        'probabilidad_si_churn': round(probabilidades[0][1], 4)
    }
    
    return jsonify(resultado)

# 4. Iniciar el servidor cuando se ejecuta el script
if __name__ == '__main__':
    # El servidor estará disponible en http://127.0.0.1:5000
    app.run(debug=True, port=5000)

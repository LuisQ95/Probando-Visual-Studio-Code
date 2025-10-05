# test_api.py
import requests
import json

# La URL de tu endpoint de predicción
url = 'http://127.0.0.1:5000/predict'

# Los datos del cliente que quieres predecir.
# Deben coincidir con las columnas que espera tu modelo.
cliente_data = {
    'edad': 45,
    'saldo_cuenta': 500,       # Saldo bajo
    'antiguedad_cliente_meses': 6, # Poca antiguedad
    'num_productos': 1,        # Solo un producto
    'tiene_tarjeta_credito': 1,
    'es_miembro_activo': 0,    # Inactivo
    'salario_estimado': 75000
}

# Hacemos la petición POST a la API
response = requests.post(url, json=cliente_data)

# Verificamos que la petición fue exitosa (código 200)
if response.status_code == 200:
    # Imprimimos el resultado de la predicción
    resultado = response.json()
    print("Petición exitosa. Resultado de la predicción:")
    print(json.dumps(resultado, indent=4))
else:
    print(f"Error al llamar a la API. Código de estado: {response.status_code}")
    print("Respuesta del servidor:", response.text)


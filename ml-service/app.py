"""
MÓDULO DE INFERENCIA Y SERVICIO API REST (app.py)

Una vez que el script 'train.py' hizo el trabajo pesado de descubrir patrones y generar el 
"cerebro" matemático (los archivos .pkl), necesitamos una forma de comunicarnos con él. 
Este script actúa como el "mesero" del restaurante: levanta un servidor web (API) que escucha 
peticiones del exterior, consulta los modelos entrenados y devuelve una respuesta. Sin este 
archivo, tu modelo entrenado se quedaría encerrado e inútil en tu disco duro.

IMPORTANCIA PARA EL ENCARGO Y EL DESARROLLO:
Es el corazón de la orquestación End-to-End, ya que permite aislar el entrenamiento de la predicción, posibilitando que 
otras aplicaciones (como tu Dashboard) consuman inteligencia artificial en tiempo real.

FLUJO OPERATIVO Y CÓMO FUNCIONA:
1. CARGA EN MEMORIA (Load): Al iniciar, el servidor carga los archivos estáticos y 
   los modelos serializados (.pkl) en la memoria RAM para responder ultra rápido.
2. ENDPOINT RAÍZ ('/'): Ruta básica de diagnóstico (Health Check) para comprobar 
   rápidamente en el navegador que el contenedor no está caído.
3. ENDPOINT DATA ('/dashboard-data'): Ruta GET que empaqueta los resultados del 
   entrenamiento en formato JSON para que el frontend (Streamlit) pueda construir gráficas.
4. ENDPOINT INFERENCIA ('/predict'): Ruta POST que recibe datos de un usuario "nuevo". 
   Aplica la estandarización exacta que aprendió el modelo (scaler.transform) y predice 
   su cluster (modelo.predict) en fracciones de segundo.

QUÉ COSAS CREA/REALIZA:
- NO entrena datos. Su función es 100% de lectura (Read) e inferencia (Predict).
- Crea una instancia de FastAPI capaz de procesar múltiples peticiones por segundo.
"""

import pandas as pd
import json
import pickle
from fastapi import FastAPI

# Inicialización de la aplicación FastAPI
app = FastAPI(title="Servicio Segmentación Usuarios Streaming")

# ==========================================
# 1. CARGA DE ARTEFACTOS (Variables Globales)
# ==========================================
# Carga la data que fue guardada en el entrenamiento
data = pd.read_csv("data/usuarios_segmentados.csv")

# Carga el modelo K-Means y el Escalador
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Carga las métricas del entrenamiento
with open("models/metricas.json") as f:
    metricas = json.load(f)

# ==========================================
# 2. DEFINICIÓN DE RUTAS (Endpoints)
# ==========================================

@app.get("/")
def inicio():
    """Ruta de diagnóstico para verificar que la API está viva."""
    return {
        "mensaje": "Servicio ML funcionando"
    }

@app.get("/dashboard-data")
def dashboard_data():
    """Provee los datos procesados al Dashboard para su visualización."""
    usuarios = pd.read_csv("data/usuarios_segmentados.csv")
    centroides = pd.read_csv("data/centroides.csv")

    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas
    }

@app.post("/predict")
def predict(datos: dict):
    """Recibe datos de un nuevo usuario, los escala y predice su segmento."""
    # Transforma el diccionario de entrada en un DataFrame de 1 fila
    data_nueva = pd.DataFrame([datos])
    
    # Escala los datos usando el MISMO escalador del entrenamiento
    X = scaler.transform(data_nueva)
    
    # Realiza la predicción (determinar el cluster)
    cluster = modelo.predict(X)

    return {"cluster": int(cluster[0])}
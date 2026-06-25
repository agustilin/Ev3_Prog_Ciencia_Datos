"""
MÓDULO DE INFERENCIA Y SERVICIO API REST (app.py)

Este módulo implementa el servidor web de producción utilizando FastAPI. 
Su función exclusiva es actuar como capa operativa de consumo (servir el modelo),
aislando por completo el proceso pesado de entrenamiento de la etapa de predicción.

Importancia Arquitectónica (MLOps):
    Permite la orquestación End-to-End de la solución. Al exponer endpoints HTTP,
    cualquier aplicación externa (como un Dashboard en Streamlit) puede consumir
    la inteligencia del modelo K-Means en tiempo real de manera desacoplada.

Flujo Operativo:
    1. Eager Loading: Al arrancar el contenedor, carga los artefactos (.pkl, .json)
       directamente en la memoria RAM para garantizar respuestas de baja latencia.
    2. Health Check (/): Endpoint de diagnóstico para verificar el estado de la API.
    3. Data Provider (/dashboard-data): Serializa los resultados históricos para el frontend.
    4. Real-time Inference (/predict): Recibe datos crudos, los estandariza y predice.
"""

import pandas as pd
import json
import pickle
from fastapi import FastAPI

# Inicialización de la aplicación FastAPI con metadatos profesionales
app = FastAPI(
    title="Servicio de Segmentación de Usuarios - Streaming",
    description="API REST de producción para la inferencia y servicio de clústeres usando K-Means.",
    version="1.0.0"
)

# ==============================================================================
# 1. CARGA DE ARTEFACTOS ESTÁTICOS (Eager Loading en Memoria RAM)
# ==============================================================================
# Nota Senior: Los archivos se cargan fuera de las funciones (en el scope global) 
# para que ocurra una sola vez al iniciar el servidor, evitando leer el disco en cada petición.

try:
    # Carga de la base de datos histórica con los clústeres ya asignados
    data = pd.read_csv("data/usuarios_segmentados.csv")

    # Deserialización del modelo y el escalador matemático en modo lectura binaria ('rb')
    modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))

    # Carga de las métricas de validación (Inercia, Silhouette Score, K-Óptimo)
    with open("models/metricas.json", "r") as f:
        metricas = json.load(f)

except FileNotFoundError as e:
    print(f"ERROR CRÍTICO: No se encontraron los artefactos en las carpetas /data o /models. "
          f"Asegúrate de ejecutar 'train.py' antes de levantar la API. Detalles: {e}")

# ==============================================================================
# 2. DEFINICIÓN DE RUTAS Y ENDPOINTS HTTP
# ==============================================================================

@app.get("/")
def inicio():
    """
    Endpoint de Diagnóstico (Health Check).
    
    Verifica de manera rápida y sin consumo de recursos que el contenedor de Docker 
    y el servicio web de FastAPI están vivos y respondiendo peticiones.
    
    Returns:
        dict: Un diccionario JSON con el estado actual del servicio.
    """
    return {
        "mensaje": "Servicio de Inferencia ML funcionando correctamente"
    }


@app.get("/dashboard-data")
def dashboard_data():
    """
    Proveedor de Datos de Entrada para el Frontend (Dashboard Data Provider).
    
    Lee los resultados del almacenamiento y los empaqueta junto con las métricas 
    del modelo en un formato JSON estructurado. Está diseñado para ser consumido 
    exclusivamente por la aplicación visual (Streamlit).
    
    Returns:
        dict: JSON con tres llaves principales:
            - 'usuarios': Lista de registros de clientes con su clúster asignado.
            - 'centroides': Coordenadas de los centros de cada grupo.
            - 'metricas': Puntuaciones de rendimiento del entrenamiento matemático.
    """
    # Se vuelven a leer para asegurar que reflejen la última ejecución de train.py
    usuarios = pd.read_csv("data/usuarios_segmentados.csv")
    centroides = pd.read_csv("data/centroides.csv")

    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas
    }


@app.post("/predict")
def predict(datos: dict):
    """
    Endpoint de Inferencia en Tiempo Real (Real-time Inference).
    
    Recibe las características sociodemográficas y de consumo de un usuario nuevo,
    aplica las transformaciones matemáticas requeridas y predice su perfil comercial.
    
    Args:
        datos (dict): Diccionario que representa un JSON enviado por el cliente.
            Debe contener las llaves exactas con las que el modelo fue entrenado 
            (ej. {'edad': 25, 'ingresos': 450000, ...}).
            
    Regla de Oro MLOps:
        Se utiliza 'scaler.transform()' y JAMÁS 'fit_transform()'. Esto garantiza
        que el nuevo usuario sea escalado usando exactamente la misma media y 
        desviación estándar que se aprendieron del conjunto de datos de entrenamiento.
        
    Returns:
        dict: JSON con el ID numérico del clúster asignado al usuario.
    """
    # 1. Convertir el JSON/Diccionario de entrada en una estructura matricial de Pandas (1 fila)
    data_nueva = pd.DataFrame([datos])
    
    # 2. Estandarizar las magnitudes numéricas para evitar sesgos en el cálculo de distancias
    X = scaler.transform(data_nueva)
    
    # 3. Calcular las distancias a los centroides y asignar el clúster más cercano
    cluster_predicho = modelo.predict(X)

    # Nota Técnica: Convertimos el output a int() nativo de Python ya que los tipos 
    # int32/int64 de NumPy no son nativamente serializables a JSON por FastAPI.
    return {
        "usuario_recibido": datos,
        "cluster_asignado": int(cluster_predicho[0])
    }
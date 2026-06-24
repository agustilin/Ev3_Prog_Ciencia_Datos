"""
MÓDULO DE ENTRENAMIENTO Y PROCESAMIENTO BATCH (train.py)

Tras definir el entorno y las dependencias (requirements.txt), necesitamos generar la "inteligencia" 
del sistema. La API (app.py) no puede existir ni responder peticiones si antes no existen los 
modelos matemáticos ya entrenados. Este script actúa como nuestro laboratorio: se ejecuta una 
sola vez (o de forma periódica en un entorno real) para descubrir los patrones y guardar los resultados.

IMPORTANCIA PARA EL ENCARGO:
Este script cumple con el núcleo analítico de la evaluación. Demuestra la capacidad de orquestar 
fuentes de datos mixtas (archivos planos + bases de datos relacionales en Docker), aplicar 
técnicas de Machine Learning no supervisado (K-Means) de forma automatizada (Método del Codo 
con kneed) y preparar visualizaciones avanzadas (PCA) para el Dashboard final.

FLUJO OPERATIVO Y CÓMO FUNCIONA:
1. EXTRACCIÓN (Extract): Lee datos estáticos de un CSV y se conecta mediante SQLAlchemy a 
   PostgreSQL (levantado en Docker) para extraer perfiles de usuario.
2. TRANSFORMACIÓN (Transform): Integra (merge) ambas fuentes usando el 'id_cliente'. 
   Elimina identificadores y estandariza las escalas de las variables (StandardScaler).
3. MODELADO (Model): Itera sobre posibles valores de K (2 a 10) calculando la inercia. 
   Usa KneeLocator para encontrar matemáticamente el punto de quiebre (K óptimo) y entrena 
   el modelo K-Means definitivo.
4. REDUCCIÓN DE DIMENSIONALIDAD: Aplica Análisis de Componentes Principales (PCA) para 
   comprimir la información en 2 dimensiones, facilitando su futura graficación.
5. CARGA/EXPORTACIÓN (Load/Save): Empaqueta todo el trabajo en archivos serializados y tablas.

ARTEFACTOS CREADOS:
- data/usuarios_data.csv        : Dataset integrado crudo.
- data/usuarios_segmentados.csv : Dataset final con etiquetas de clusters y coordenadas PCA.
- data/centroides.csv           : Coordenadas originales de los centros de cada segmento.
- models/metricas.json          : Registro de métricas (K óptimo, Silhouette, Varianza PCA).
- models/modelo_kmeans.pkl      : Modelo de agrupamiento entrenado.
- models/scaler.pkl             : Objeto escalador para normalizar nuevos datos en la API.
- models/pca.pkl                : Modelo PCA para reducir nuevos datos en la API.
"""

import pandas as pd
import json
import pickle
import os

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA

# Crea la carpeta de modelos si no existe
os.makedirs("models", exist_ok=True)

# Archivo CSV con datos de streaming
usuarios_streaming = pd.read_csv("data/usuarios_streaming.csv")

# Fuente desde la BD PostgreSQL
engine = create_engine("postgresql://admin:admin@postgres:5432/streaming_usuarios")

perfil_usuario = pd.read_sql(
    """
    SELECT *
    FROM perfil_usuario
    """,
    engine
)

# Integración: merge de ambas fuentes de datos
data = usuarios_streaming.merge(perfil_usuario, on="id_cliente")

# Guarda el archivo con la data integrada
data.to_csv("data/usuarios_data.csv", index=False)

# Variables del modelo
X = data.drop(columns=["id_cliente"])

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouettes = []
for k in range(2, 11):
    modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
    modelo.fit(X_scaled)

    inertias.append(modelo.inertia_)
    silhouettes.append(silhouette_score(X_scaled, modelo.labels_))

# Búsqueda del K óptimo automatizada
kl = KneeLocator(
    range(2, 11),
    inertias,
    curve='convex',
    direction='decreasing'
)

# Modelo Final
k_optimo = kl.elbow
kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)

# Predicciones
clusters = kmeans.fit_predict(X_scaled)
data["cluster"] = clusters

print("Modelo de segmentación creado!!!")

# Reducción de dimensionalidad para visualización
pca = PCA(n_components=2)
componentes = pca.fit_transform(X_scaled)

data["pc1"] = componentes[:, 0]
data["pc2"] = componentes[:, 1]

# Guarda data con los cluster y dos componentes principales
data.to_csv("data/usuarios_segmentados.csv", index=False)

# Guarda las métricas 
metricas = {
    "k_optimo": int(k_optimo),
    "silhouette_score": float(silhouette_score(X_scaled, data["cluster"])),
    "n_usuarios": int(len(data)),
    "n_clusters": int(k_optimo),
    "varianza_pca": float(
        pca.explained_variance_ratio_.sum()
    )
}

with open("models/metricas.json", "w") as f:
    json.dump(metricas, f, indent=4)

# Guarda los centroides
centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)

centroides_df = pd.DataFrame(
    centroides_original,
    columns=X.columns
)

centroides_df.to_csv("data/centroides.csv", index=False)

# Guardar modelo y data escalada (Serialización)
pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(pca, open("models/pca.pkl", "wb"))

print("Modelo guardado y proceso finalizado con éxito.")


# docker-compose up --build postgres ml-service (este comando creará la imagen y levantará el contenedor ml-service antes de entrar al dashboard, para validar que todo funciona)
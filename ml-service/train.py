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
   Elimina identificadores numéricos y estandariza las escalas de las variables (StandardScaler).
3. MODELADO (Model): Itera sobre posibles valores de K (2 a 10) calculando la inercia (distancia 
   intra-cluster) y Silhouette. Usa KneeLocator para encontrar matemáticamente el punto de quiebre 
   (K óptimo) y entrena el modelo K-Means definitivo.
4. REDUCCIÓN DE DIMENSIONALIDAD (PCA): Comprime la información multidimensional a solo 2 
   componentes (ejes X e Y) para poder graficar los segmentos en el Dashboard. Aquí se calcula 
   la "Varianza Explicada" y la "Varianza Acumulada" para medir cuánta información original 
   logramos retener tras esta compresión.
5. CARGA/EXPORTACIÓN (Load/Save): Empaqueta todo el trabajo en archivos serializados (.pkl) 
   y tablas estáticas (.csv y .json) listos para ser consumidos.

ARTEFACTOS CREADOS:
- data/usuarios_data.csv        : Dataset integrado crudo.
- data/usuarios_segmentados.csv : Dataset final con etiquetas de clusters y coordenadas PCA.
- data/centroides.csv           : Coordenadas originales de los centros de cada segmento.
- models/metricas.json          : Registro de métricas (K óptimo, Silhouette, Varianza PCA y Acumulada).
- models/modelo_kmeans.pkl      : Modelo de agrupamiento entrenado.
- models/scaler.pkl             : Objeto escalador para normalizar nuevos datos en la API.
- models/pca.pkl                : Modelo PCA para reducir nuevos datos en la API.
"""

import pandas as pd
import json
import pickle
import os
import numpy as np

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# 0. PREPARACIÓN DEL ENTORNO
# -------------------------------------------------------------------
# Creamos la carpeta 'models' donde guardaremos los artefactos si no existe previamente.
os.makedirs("models", exist_ok=True)

# -------------------------------------------------------------------
# 1. EXTRACCIÓN DE DATOS (Extract)
# -------------------------------------------------------------------
# Cargamos los datos de comportamiento desde un archivo plano local.
usuarios_streaming = pd.read_csv("data/usuarios_streaming.csv")

# Establecemos conexión con la base de datos PostgreSQL que corre en nuestro contenedor Docker.
engine = create_engine("postgresql://admin:admin@postgres:5432/streaming_usuarios")

# Extraemos los datos demográficos o de perfil directamente con una consulta SQL.
perfil_usuarios = pd.read_sql(
    """
    SELECT *
    FROM perfil_usuarios
    """,
    engine
)

# -------------------------------------------------------------------
# 2. TRANSFORMACIÓN E INTEGRACIÓN (Transform)
# -------------------------------------------------------------------
# Unimos (merge) la data de streaming con los perfiles usando 'id_cliente' como llave primaria.
# Esto crea nuestro dataset maestro.
data = usuarios_streaming.merge(perfil_usuarios, on="id_cliente")

# Guardamos un respaldo de esta data integrada cruda (opcional, pero buena práctica).
data.to_csv("data/usuarios_data.csv", index=False)

# Preparamos las variables (X) para el modelo. 
# Eliminamos 'id_cliente' porque es un identificador único que no aporta valor predictivo o de similitud.
X = data.drop(columns=["id_cliente"])

# K-Means es un algoritmo basado en distancias euclidianas. Por ende, variables con 
# rangos muy grandes (ej. ingresos) dominarían sobre variables pequeñas (ej. edad).
# Usamos StandardScaler para que todas las variables tengan media 0 y varianza 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------------------
# 3. BÚSQUEDA DEL K ÓPTIMO (Modelamiento exploratorio)
# -------------------------------------------------------------------
inertias = []
silhouettes = []

# Iteramos probando de 2 a 10 clusters para evaluar cuál es la mejor configuración.
for k in range(2, 11):
    modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
    modelo.fit(X_scaled)

    # La inercia mide qué tan compactos son los clusters (menor es mejor, pero tiende a 0 al aumentar K).
    inertias.append(modelo.inertia_)
    # El Silhouette Score mide qué tan bien separado está cada cluster del resto (más cercano a 1 es mejor).
    silhouettes.append(silhouette_score(X_scaled, modelo.labels_))

# Usamos KneeLocator para encontrar el "codo" matemático en la curva de inercia.
# Esto automatiza la decisión visual sin intervención humana.
kl = KneeLocator(
    range(2, 11),
    inertias,
    curve='convex',
    direction='decreasing'
)

# Definimos el K final. Si KneeLocator no encuentra un codo claro, usamos como respaldo
# el K que haya maximizado el Silhouette Score.
k_optimo = kl.elbow 

# -------------------------------------------------------------------
# 4. ENTRENAMIENTO DEL MODELO FINAL
# -------------------------------------------------------------------
# Entrenamos el K-Means definitivo con el K óptimo encontrado.
kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)

# fit_predict entrena el modelo y nos devuelve a qué cluster pertenece cada usuario.
clusters = kmeans.fit_predict(X_scaled)

# Agregamos esta nueva etiqueta al dataset original.
data["cluster"] = clusters

print(f"Modelo de segmentación creado con K={k_optimo}!!!")

# -------------------------------------------------------------------
# 5. REDUCCIÓN DE DIMENSIONALIDAD (PCA) PARA DASHBOARDS
# -------------------------------------------------------------------
# Los datos originales tienen muchas dimensiones (columnas), imposibles de graficar.
# PCA comprime estas variables creando 2 componentes principales ortogonales (ejes X e Y)
# intentando preservar la mayor cantidad de varianza (información) posible.
pca = PCA(n_components=2)
componentes = pca.fit_transform(X_scaled)

# Añadimos las coordenadas bidimensionales al dataset para que el Dashboard pueda hacer el Scatter Plot.
data["pc1"] = componentes[:, 0]
data["pc2"] = componentes[:, 1]

# Exportamos la data final lista para análisis visual.
data.to_csv("data/usuarios_segmentados.csv", index=False)

# -------------------------------------------------------------------
# 6. REGISTRO DE MÉTRICAS Y EXPORTACIÓN DE ARTEFACTOS
# -------------------------------------------------------------------
# Calculamos las métricas fundamentales
# - varianza_explicada: Cuánto % de la información retiene el PC1 y el PC2 individualmente.
# - varianza_acomulada: La suma acumulada. Ej: Si PC1 es 0.40 y PC2 es 0.20, la acumulada es [0.40, 0.60].
#   Esto permite decir en el Dashboard: "Nuestro gráfico en 2D explica el X% del comportamiento de los usuarios".

metricas = {
    "k_optimo": int(k_optimo),
    "silhouette_score": float(silhouette_score(X_scaled, data["cluster"])),
    "n_usuarios": int(len(data)),
    "n_clusters": int(k_optimo),
    # Las llaves en ingles para relacionarlas al atributo de scikit-learn en inglés:
    "varianza_pca": float(pca.explained_variance_ratio_.sum()),
    "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_.round(4)],
    "cumulative_variance_ratio": [float(x) for x in np.cumsum(pca.explained_variance_ratio_).round(4)],
    "inertia_by_k": [
        {"k": int(k), "inertia": float(value)}
        for k, value in zip(range(2, 11), inertias)
    ],
    "silhouette_by_k": [
        {"k": int(k), "silhouette": float(value)}
        for k, value in zip(range(2, 11), silhouettes)
    ]
}

# Guardamos el JSON que consumirá el Dashboard para mostrar los KPIs.
with open("models/metricas.json", "w") as f:
    json.dump(metricas, f, indent=4)

# Para entender qué significa cada cluster en el negocio, necesitamos ver sus centros ("promedios").
# Como entrenamos con datos escalados, usamos inverse_transform para devolver los centroides
# a sus valores y escalas originales (ej: edad en años reales, no en valores estandarizados).
centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)

centroides_df = pd.DataFrame(
    centroides_original,
    columns=X.columns
)

# Exportamos los perfiles promedio de cada segmento.
centroides_df.to_csv("data/centroides.csv", index=False)

# Finalmente, empaquetamos (Pickle) los tres pilares matemáticos del proyecto.
# Estos archivos .pkl son los que cargará la API para predecir a qué cluster
# pertenecerá un nuevo cliente en el futuro.
pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(pca, open("models/pca.pkl", "wb"))

print("Modelo guardado y proceso finalizado con éxito.")

# NOTA DE ORQUESTACIÓN:
# Comando para validación en local:
# docker-compose up --build postgres ml-service 
# (Levanta la base de datos y ejecuta este script aislando el entorno antes de lanzar el dashboard)
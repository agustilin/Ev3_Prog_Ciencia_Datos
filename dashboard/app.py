import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
Dashboard de Analítica y Segmentación de Usuarios de Streaming

Este módulo implementa una interfaz web interactiva utilizando Streamlit para visualizar,
justificar y perfilar la segmentación de clientes de una plataforma de streaming. El dashboard 
consume datos de un servicio de Machine Learning remoto (`ml-service`) y expone la información 
en tres bloques estratégicos:
1. Justificación técnica del modelo: Evaluación del número óptimo de clústeres mediante métricas 
   de Inercia (Método del Codo), coeficiente de Silhouette y varianza explicada por PCA.
2. Distribución de segmentos: Visualización espacial en componentes principales y volumen de usuarios.
3. Perfilamiento de clústeres: Análisis detallado del comportamiento de los usuarios empleando 
   las variables originales del negocio mediante tablas estadísticas, boxplots y un mapa de calor.

El objetivo principal es transformar datos analíticos complejos en insights accionables para la 
toma de decisiones comerciales, retención de clientes y personalización de campañas de marketing.
"""

# Configuración inicial de la página de Streamlit (Layout extendido y título del navegador)
st.set_page_config(
    page_title="Dashboard Segmentación Streaming",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Títulos principales y descripción inicial en la interfaz de usuario
st.title("Dashboard Analítico de Segmentación de Usuarios")
st.markdown(
    "Este dashboard presenta la justificación técnica del modelo, la visualización de segmentos y el perfilamiento de clústeres "
    "con base en las variables originales del dataset antes del PCA."
)

# Función optimizada con caché para realizar la petición HTTP al servicio de ML y extraer los datos
@st.cache_data(show_spinner=False)
def cargar_datos():
    respuesta = requests.get("http://ml-service:8000/dashboard-data")
    respuesta.raise_for_status()
    payload = respuesta.json()
    usuarios = pd.DataFrame(payload["usuarios"])
    metricas = payload["metricas"]
    centroides = pd.DataFrame(payload["centroides"])
    return usuarios, metricas, centroides

# Ejecución de la carga de datos de usuarios, métricas de rendimiento y centroides
usuarios, metricas, centroides = cargar_datos()

# Definición de la lista de variables operacionales y demográficas originales para el perfilamiento
variables_originales = [
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km"
]

# Preparación de datos: cálculo de promedios por clúster y distribución porcentual del volumen de usuarios
grupo_cluster = usuarios.groupby("cluster")
perfil_promedios = grupo_cluster[variables_originales].mean().round(2)
conteos_cluster = usuarios["cluster"].value_counts().sort_index()
porcentaje_cluster = (conteos_cluster / conteos_cluster.sum() * 100).round(1)


# Funciones auxiliares de maquetación para centrar textos, markdowns y gráficos usando columnas de Streamlit
def center_text(title: str, subtitle: str = None):
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown(f"<h2 style='text-align:center'>{title}</h2>", unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<p style='text-align:center'>{subtitle}</p>", unsafe_allow_html=True)


def center_markdown(text: str):
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown(f"<p style='text-align:center'>{text}</p>", unsafe_allow_html=True)


def center_plot(fig):
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)


# SECCIÓN 1: Presentación del encabezado para la Justificación del Modelo
center_text(
    "1. Justificación del Modelo",
    "Este primer bloque presenta las métricas de evaluación y la evidencia de por qué el modelo de segmentación es consistente."
)

# Construcción y renderizado del gráfico de barras y líneas para evaluar la Varianza Explicada y Acumulada (PCA)
df_pca = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(len(metricas["explained_variance_ratio"]))],
    "Varianza Explicada": metricas["explained_variance_ratio"],
    "Varianza Acumulada": metricas["cumulative_variance_ratio"]
})

fig_varianza = go.Figure()
fig_varianza.add_trace(
    go.Bar(
        x=df_pca["Componente"],
        y=df_pca["Varianza Explicada"],
        name="Varianza explicada",
        marker_color="#005b96"
    )
)
fig_varianza.add_trace(
    go.Scatter(
        x=df_pca["Componente"],
        y=df_pca["Varianza Acumulada"],
        name="Varianza acumulada",
        mode="lines+markers",
        marker=dict(color="#ff7f0e")
    )
)
fig_varianza.update_layout(
    title="Varianza explicada y acumulada por componente PCA",
    xaxis_title="Componente principal",
    yaxis_title="Proporción de varianza",
    legend_title="Métrica",
    template="plotly_white"
)

center_plot(fig_varianza)

# Despliegue de los principales KPIs del modelo (K óptimo, Silhouette Score y Varianza Acumulada Total)
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("K óptimo", metricas["k_optimo"])
    with c2:
        st.metric("Silhouette", f"{metricas['silhouette_score']:.3f}")
    with c3:
        st.metric("Varianza acumulada PCA", f"{metricas['cumulative_variance_ratio'][-1] * 100:.1f}%")

center_markdown(
    "La varianza explicada muestra cuánto aporta cada componente principal al modelo. "
)
center_markdown(
    "Beneficio para el negocio: permite visualizar y comunicar claramente la calidad del modelo, dando confianza para usar estos clusters en decisiones comerciales."
)

# Construcción y visualización conjunta de la Curva del Codo (Inercia) y la curva del índice de Silhouette por cada K
df_elbow = pd.DataFrame(metricas["inertia_by_k"])
df_silhouette = pd.DataFrame(metricas["silhouette_by_k"])

fig_curvas = go.Figure()
fig_curvas.add_trace(
    go.Scatter(
        x=df_elbow["k"],
        y=df_elbow["inertia"],
        mode="lines+markers",
        name="Inercia",
        line=dict(color="#1f77b4"),
        marker=dict(size=10)
    )
)
fig_curvas.add_trace(
    go.Scatter(
        x=df_silhouette["k"],
        y=df_silhouette["silhouette"],
        mode="lines+markers",
        name="Silhouette",
        line=dict(color="#ff7f0e"),
        marker=dict(size=10)
    )
)
fig_curvas.add_trace(
    go.Scatter(
        x=[metricas["k_optimo"]],
        y=[df_elbow.loc[df_elbow["k"] == metricas["k_optimo"], "inertia"].iloc[0]],
        mode="markers",
        name="K seleccionado",
        marker=dict(size=14, color="#2ca02c", symbol="diamond")
    )
)
fig_curvas.update_layout(
    title="Curva del codo e índice de Silhouette para elegir K",
    xaxis_title="Número de clústeres (K)",
    yaxis_title="Valor",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

center_plot(fig_curvas)
center_markdown(
    "La curva del codo indica que a partir de K=3 la mejora en inercia es marginal, y la curva de Silhouette confirma una separación estable. "
    "Esto justifica la elección del número óptimo de clusters con evidence visual y cuantitativa."
)
center_markdown(
    "Beneficio para el negocio: asegura que el modelo no esté sobresegmentando, evitando grupos demasiado pequeños o poco accionables."
)

st.markdown("---")

# SECCIÓN 2: Presentación del encabezado para la Visualización y Distribución de los Segmentos
center_text(
    "2. Visualización y Distribución de Segmentos",
    "Aquí presentamos la separación en el espacio PCA, la distribución porcentual de cada cluster y una vista con las dos variables más relevantes."
)

# Creación del gráfico de dispersión (Scatter Plot) en el espacio bidimensional de las dos primeras componentes principales (PC1 vs PC2)
fig_scatter = px.scatter(
    usuarios,
    x="pc1",
    y="pc2",
    color=usuarios["cluster"].astype(str),
    color_discrete_sequence=px.colors.qualitative.Set1,
    hover_data=variables_originales,
    title="Visualización PCA de los segmentos",
    labels={"pc1": "PC1", "pc2": "PC2", "cluster": "Cluster"}
)
fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
fig_scatter.update_layout(
    template="plotly_white",
    xaxis=dict(showgrid=True, gridcolor="LightGrey"),
    yaxis=dict(showgrid=True, gridcolor="LightGrey"),
    legend_title_text="Cluster",
    legend=dict(title_font_size=12)
)
center_plot(fig_scatter)

center_markdown(
    "Este gráfico PCA muestra que los clusters se distribuyen de forma diferenciada en el espacio de las dos primeras componentes. "
    "La separación visual indica que los segmentos son distinguibles y que el modelo captura patrones de comportamiento reales."
)
center_markdown(
    "Beneficio para el negocio: permite validar que los segmentos identificados pueden usarse para estrategias de marketing personalizadas, retención y ofertas específicas."
)

# Generación del gráfico de torta (Donut Chart) para ilustrar la participación y distribución porcentual del total de usuarios por clúster
fig_pie = px.pie(
    names=[f"Cluster {idx}" for idx in conteos_cluster.index],
    values=conteos_cluster.values,
    title="Distribución porcentual de usuarios por clúster",
    hole=0.35,
    custom_data=[porcentaje_cluster.values],
    labels={"label": "Cluster", "value": "Usuarios"}
)
fig_pie.update_traces(
    textposition="inside",
    textinfo="percent+label",
    hovertemplate="%{label}<br>%{value} usuarios<br>%{customdata[0]}%<extra></extra>"
)
center_plot(fig_pie)
center_markdown(
    "La torta destaca el peso relativo de cada cluster tanto en porcentaje como en número de usuarios. "
    "Así podemos ver cuál es el segmento dominante y qué grupos representan oportunidades de crecimiento o riesgo."
)
center_markdown(
    "Beneficio para el negocio: facilita la priorización de recursos hacia clusters con mayor volumen o mayor potencial de valor."
)

# Creación de gráfico de barras complementario para comparar de forma directa el conteo absoluto de usuarios por segmento
bar_tamanio = pd.DataFrame({
    "Cluster": [f"Cluster {idx}" for idx in conteos_cluster.index],
    "Usuarios": conteos_cluster.values
})
fig_bar_tamanio = px.bar(
    bar_tamanio,
    x="Cluster",
    y="Usuarios",
    title="Tamaño de los grupos por cluster",
    text="Usuarios",
    labels={"Usuarios": "Número de usuarios"},
    color="Cluster",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_bar_tamanio.update_traces(textposition="outside")
fig_bar_tamanio.update_layout(template="plotly_white", showlegend=False)
center_plot(fig_bar_tamanio)
center_markdown(
    "El gráfico de barras permite comparar de forma directa el tamaño de cada cluster. "
    "Es una forma más precisa de ver el volumen relativo de usuarios en cada segmento que el gráfico de torta."
)

# Bloque interactivo: Selector de clústeres para filtrar y comparar dinámicamente las variables clave del negocio (Gasto, Consumo y Contenidos)
center_text(
    "Segmentación con las 2 características más representativas",
    None
)
cluster_seleccion = st.selectbox(
    "Selecciona un cluster para explorar métricas promedio:",
    ["Todos"] + [f"Cluster {idx}" for idx in conteos_cluster.index]
)
if cluster_seleccion == "Todos":
    datos_cluster = usuarios.copy()
else:
    seleccion_num = int(cluster_seleccion.split(" ")[1])
    datos_cluster = usuarios[usuarios["cluster"] == seleccion_num]

metricas_cluster = (
    datos_cluster[["gasto_mensual", "horas_consumo_mensual", "cantidad_contenidos_vistos"]]
    .mean()
    .round(1)
    .reset_index(name="Promedio")
    .rename(columns={"index": "Métrica"})
)
fig_interactivo = px.bar(
    metricas_cluster,
    x="Métrica",
    y="Promedio",
    title=f"Métricas promedio para {cluster_seleccion}",
    text="Promedio",
    labels={"Promedio": "Valor promedio"},
    color="Métrica",
    color_discrete_sequence=px.colors.qualitative.D3
)
fig_interactivo.update_traces(textposition="outside")
fig_interactivo.update_layout(template="plotly_white", showlegend=False)
center_plot(fig_interactivo)
center_markdown(
    "Este gráfico interactivo permite explorar en detalle el comportamiento promedio del cluster seleccionado. "
    "Permite ver si un cluster tiene mayor gasto, más horas de consumo o mayor actividad de contenido, apoyando decisiones específicas por segmento."
)

# Graficación del comportamiento real de los clústeres cruzando directamente dos variables críticas: Horas de consumo vs Gasto mensual
fig_2vars = px.scatter(
    usuarios,
    x="horas_consumo_mensual",
    y="gasto_mensual",
    color=usuarios["cluster"].astype(str),
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Clusters según horas_consumo_mensual y gasto_mensual",
    labels={
        "horas_consumo_mensual": "Horas de consumo mensual",
        "gasto_mensual": "Gasto mensual",
        "cluster": "Cluster"
    },
    hover_data=["cluster"]
)
fig_2vars.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
fig_2vars.update_layout(template="plotly_white")
center_plot(fig_2vars)

center_markdown(
    "Este gráfico con horas_consumo_mensual y gasto_mensual muestra cómo los clusters se comportan en términos de uso y valor. "
    "Los distintos colores permiten detectar rápidamente si existen segmentos de alto consumo/alto gasto, o segmentos de bajo valor."
)
center_markdown(
    "Beneficio para el negocio: identifica prioridades para retención y monetización, y ayuda a diseñar ofertas personalizadas según intensidad de uso y capacidad de gasto."
)

st.markdown("---")

# SECCIÓN 3: Presentación del encabezado para el Perfilamiento de los Clústeres con métricas de negocio tradicionales
center_text(
    "3. Perfilamiento de Clústeres",
    "En esta sección se analizan los perfiles promedio de cada clúster y se observa la dispersión del gasto mensual, la variable clave de negocio."
)

# Despliegue del DataFrame con el promedio exacto de las 15 variables de negocio para cada uno de los clústeres
center_text("Indicadores promedio por clúster", None)
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.dataframe(perfil_promedios.reset_index().rename(columns={"cluster": "Cluster"}))

# Creación de diagramas de caja (Boxplots) para analizar la variabilidad, mediana y presencia de outliers en el Gasto Mensual por clúster
center_text("Boxplot de gasto_mensual por clúster", None)
fig_box = px.box(
    usuarios,
    x="cluster",
    y="gasto_mensual",
    color=usuarios["cluster"].astype(str),
    points="outliers",
    title="Distribución de gasto mensual por clúster",
    labels={"cluster": "Cluster", "gasto_mensual": "Gasto mensual"}
)
fig_box.update_layout(template="plotly_white", showlegend=False)
center_plot(fig_box)
center_markdown(
    "El boxplot muestra la dispersión del gasto mensual dentro de cada cluster, incluyendo valores atípicos. "
    "Esto permite ver no sólo promedios, sino también la variabilidad y la presencia de usuarios con gasto inusualmente alto o bajo."
)
center_markdown(
    "Beneficio para el negocio: ayuda a identificar segmentos con alto potencial de upsell y a diseñar acciones específicas para reducir el churn en grupos de bajo gasto."
)

# Estandarización de promedios (Z-score) y visualización en un Mapa de Calor (Heatmap) interactivo para identificar fortalezas y debilidades relativas de cada segmento
center_text("Mapa de calor de promedios estandarizados", None)
promedio_cluster = perfil_promedios.copy()
scaler = StandardScaler()
promedio_escalado = scaler.fit_transform(promedio_cluster)
df_heatmap = pd.DataFrame(
    promedio_escalado,
    index=[f"Cluster {int(c)}" for c in promedio_cluster.index],
    columns=promedio_cluster.columns
)

fig_heat = px.imshow(
    df_heatmap,
    labels=dict(x="Variable", y="Cluster", color="Z-score"),
    x=df_heatmap.columns,
    y=df_heatmap.index,
    color_continuous_scale="RdBu",
    zmin=-2,
    zmax=2,
    text_auto=True
)
fig_heat.update_layout(title="Heatmap estandarizado de promedios por clúster", template="plotly_white")
center_plot(fig_heat)
center_markdown(
    "El heatmap estandarizado compara la posición relativa de cada clúster en todas las variables originales. "
    "Los valores altos y bajos se ven con claridad, lo que permite identificar rápidamente fortalezas y debilidades de cada segmento."
)
center_markdown(
    "Beneficio para el negocio: posibilita crear perfiles accionables, como un cluster de usuarios de alto gasto y alto engagement frente a uno de bajo gasto y bajo uso. "
    "Esto sirve de base para campañas de retención, propuestas de valor y segmentación de productos."
)
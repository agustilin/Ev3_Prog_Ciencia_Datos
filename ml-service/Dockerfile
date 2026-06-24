# DOCKERFILE - SERVICIO DE MACHINE LEARNING (ml-service)


# Es la guía de instrucciones paso a paso para que Docker sepa cómo armar 
# el entorno del sistema. Garantiza que el sistema funcione igual en cualquier computadora, 
# evitando el error de "en mi máquina funciona pero en la tuya no".
#
# IMPORTANCIA PARA EL ENCARGO:
# Es el eslabón final de la orquestación. Permite "empaquetar" todo el trabajo 
# de ciencia de datos (código + librerías + modelo) en una pieza de software 
# portable y lista para ejecutarse automáticamente.
#
# FLUJO OPERATIVO:
# 1. Base: Utiliza una imagen oficial de Python 3.11.
# 2. Entorno: Prepara el espacio de trabajo (/app).
# 3. Dependencias: Instala las librerías necesarias definidas en requirements.txt.
# 4. Código: Copia tus scripts (train.py, app.py) al contenedor.
# 5. Ejecución (CMD): Primero asegura que el modelo 
#    exista (train.py) y, solo después, levanta la API (uvicorn).


# 1. Usamos una imagen oficial de Python
FROM python:3.11

# 2. Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiamos el archivo de requerimientos para que Docker sepa qué instalar
COPY requirements.txt .

# 4. Instalamos todas las librerías necesarias para el proyecto
RUN pip install -r requirements.txt

# 5. Copiamos todos nuestros scripts (train.py, app.py) y carpetas al contenedor
COPY . .

# 6. Exponemos el puerto por donde hablará la API
EXPOSE 8000

# 7. EL COMANDO MAESTRO (La solución al error de archivo no encontrado):
# Primero ejecuta el entrenamiento (creando los .pkl) y, SOLO si es exitoso (&&),
# levanta inmediatamente el servidor de FastAPI.
CMD python train.py && \
    uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000
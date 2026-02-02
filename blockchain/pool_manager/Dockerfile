FROM python:3.11-slim

WORKDIR /app

# Copiamos el paquete del pool manager
COPY pool_manager ./pool_manager

# Copiamos config compartida
COPY config ./config

# Instalamos dependencias
RUN pip install --no-cache-dir \
    flask \
    pika \
    redis \
    requests \
    python-dotenv

# Exponemos el puerto del Pool Manager
EXPOSE 6000

# Ejecutamos como módulo (imports correctos)
CMD ["python", "-m", "pool_manager.pool_manager"]

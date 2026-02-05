FROM python:3.11-slim

WORKDIR /app

# Copiamos el paquete del pool manager
COPY . .

# Instalamos dependencias
RUN pip install --no-cache-dir \
    flask \
    pika \
    redis \
    requests \
    python-dotenv

# Ejecutamos como módulo (imports correctos)
CMD ["python", "pool_manager.py"]

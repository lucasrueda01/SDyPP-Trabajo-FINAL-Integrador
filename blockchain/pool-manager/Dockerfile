FROM python:3.11-slim

WORKDIR /app

# Copiamos el código del pool manager
COPY pool-manager/pool-manager.py .

# Copiamos settings y config compartida
COPY config ./config

# Dependencias necesarias
RUN pip install --no-cache-dir \
    flask \
    pika \
    redis \
    python-dotenv

# Comando de arranque
CMD ["python", "pool_manager.py"]

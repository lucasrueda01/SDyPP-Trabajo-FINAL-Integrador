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
    python-dotenv \
    prometheus_client \
    gunicorn

# Ejecutamos como módulo (imports correctos)
CMD ["gunicorn", "-w", "1", "--threads", "1", "-b", "0.0.0.0:5000", "pool_manager:app"]

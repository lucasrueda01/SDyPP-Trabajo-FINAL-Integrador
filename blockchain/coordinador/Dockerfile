FROM python:3.11-slim

WORKDIR /app

# Copiamos todo el contenido del servicio
COPY . .

# Instalamos dependencias
RUN pip install --no-cache-dir \
    flask \
    pika \
    redis \
    google-cloud-storage \
    python-dotenv \
    prometheus_client \
    flask-cors

# Comando de arranque
CMD ["python", "coordinador.py"]
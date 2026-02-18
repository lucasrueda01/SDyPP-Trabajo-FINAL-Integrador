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
    flask-cors \
    gunicorn

# Comando de arranque. Usamos Gunicorn para paralelizar el servicio y mejorar su rendimiento. 
CMD ["gunicorn", "-w", "1", "--threads", "1", "-b", "0.0.0.0:5000", "coordinador:app"]
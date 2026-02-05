FROM python:3.11-slim

WORKDIR /app

# Copiamos todo el contenido del worker CPU
COPY . .

RUN pip install --no-cache-dir \
    pika \
    requests \
    python-dotenv

# Ejecutamos el worker directamente
CMD ["python", "worker_cpu.py"]

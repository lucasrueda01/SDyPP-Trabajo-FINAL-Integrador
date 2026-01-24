FROM python:3.11-slim

WORKDIR /app

COPY coordinador/coordinador.py .
COPY config ./config

RUN pip install flask pika redis google-cloud-storage python-dotenv

CMD ["python", "coordinador.py"]

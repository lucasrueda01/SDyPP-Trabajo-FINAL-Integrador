FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    redis \
    google-cloud-compute \
    python-dotenv \
    prometheus_client

COPY . .

CMD ["python", "scaler.py"]

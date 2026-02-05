FROM python:3.11-slim

WORKDIR /app

RUN pip install redis google-cloud-compute python-dotenv

# config compartido
COPY config /app/config

# startup script (solo se empaqueta)
COPY infraestructura/terraform/templates/worker_cpu_startup.sh \
     /app/worker_cpu_startup.sh

# scaler
COPY cpu-scaler/scaler.py /app/

CMD ["python", "scaler.py"]

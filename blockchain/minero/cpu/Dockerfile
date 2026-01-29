FROM python:3.11-slim

WORKDIR /app

COPY minero ./minero
COPY config ./config

RUN pip install pika requests python-dotenv

CMD ["python", "-m", "minero.cpu.worker_cpu"]

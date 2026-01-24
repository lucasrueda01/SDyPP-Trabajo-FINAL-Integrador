FROM python:3.11-slim

WORKDIR /app

COPY minero/worker_cpu.py .
COPY config ./config

RUN pip install pika requests python-dotenv

CMD ["python", "worker_cpu.py"]

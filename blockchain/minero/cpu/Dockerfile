FROM python:3.11-slim

WORKDIR /app

COPY minero/cpu/__init__.py /app/minero/cpu/__init__.py
COPY minero/cpu/worker_cpu.py /app/minero/cpu/worker_cpu.py
COPY config /app/config

RUN pip install pika requests python-dotenv

CMD ["python", "-m", "minero.cpu.worker_cpu"]

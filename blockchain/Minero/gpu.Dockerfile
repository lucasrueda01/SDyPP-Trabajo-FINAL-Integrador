FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

# -----------------------
# Sistema y Python
# -----------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip build-essential \
 && rm -rf /var/lib/apt/lists/*

# -----------------------
# Código Python (paquetes)
# -----------------------
COPY minero/__init__.py /app/minero/__init__.py
COPY minero/worker_gpu.py /app/minero/worker_gpu.py
COPY minero/minero_gpu.py /app/minero/minero_gpu.py

# Código CUDA
COPY minero/minero_cuda.cu /app/minero/minero_cuda.cu
COPY minero/md5.cu /app/minero/md5.cu

# Configuración compartida
COPY config /app/config

# -----------------------
# Compilación CUDA (build time)
# -----------------------
RUN nvcc /app/minero/minero_cuda.cu -o /app/minero/minero_cuda -allow-unsupported-compiler \
 && chmod +x /app/minero/minero_cuda

# -----------------------
# Dependencias Python
# -----------------------
RUN pip3 install pika requests python-dotenv

# -----------------------
# Run
# -----------------------
CMD ["python3", "-m", "minero.worker_gpu"]

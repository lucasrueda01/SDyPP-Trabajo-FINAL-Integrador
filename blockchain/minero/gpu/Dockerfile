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
COPY minero/gpu/__init__.py /app/minero/gpu/__init__.py
COPY minero/gpu/worker_gpu.py /app/minero/gpu/worker_gpu.py
COPY minero/gpu/minero_gpu.py /app/minero/gpu/minero_gpu.py

# Código CUDA
COPY minero/gpu/minero_cuda.cu /app/minero/gpu/minero_cuda.cu
COPY minero/gpu/md5.cu /app/minero/gpu/md5.cu

# Configuración compartida
COPY config /app/config

# -----------------------
# Compilación CUDA (build time)
# -----------------------
RUN nvcc /app/minero/gpu/minero_cuda.cu -o /app/minero/gpu/minero_cuda -allow-unsupported-compiler \
 && chmod +x /app/minero/gpu/minero_cuda
 
# -----------------------
# Dependencias Python
# -----------------------
RUN pip3 install pika requests python-dotenv

# -----------------------
# Run
# -----------------------
CMD ["python3", "-m", "minero.gpu.worker_gpu"]
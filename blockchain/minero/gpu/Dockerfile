FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

# -----------------------
# Sistema y Python
# -----------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# -----------------------
# Copiamos TODO el worker GPU
# (contexto = blockchain/minero/gpu)
# -----------------------
COPY . .

# -----------------------
# Compilación CUDA (build time)
# -----------------------
RUN nvcc minero_cuda.cu \
    -o minero_cuda \
    -allow-unsupported-compiler \
 && chmod +x minero_cuda

# -----------------------
# Dependencias Python
# -----------------------
RUN pip3 install --no-cache-dir \
    pika \
    requests \
    python-dotenv

# -----------------------
# Run
# -----------------------
CMD ["python3", "worker_gpu.py"]

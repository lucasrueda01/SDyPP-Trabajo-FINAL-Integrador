#!/bin/bash
set -e

echo "Inicializando worker CPU..."

# -----------------------------
# Variables de entorno (HARDCODEADAS)
# -----------------------------
export COORDINADOR_HOST="146.148.44.118"
export COORDINADOR_PORT="5000"

export RABBIT_HOST="34.42.128.134"
export RABBIT_USER="blockchain"
export RABBIT_PASSWORD="blockchain123"
export RABBIT_PORT="5672"
export RABBIT_VHOST="blockchain"

export POOL_MANAGER_HOST="34.67.190.185"
export POOL_MANAGER_PORT="6000"

export CPU_CAPACITY="10"
export HEARTBEAT_TTL="10"

# -----------------------------
# Dependencias
# -----------------------------
apt-get update
apt-get install -y docker.io
systemctl enable --now docker

# -----------------------------
# Ejecutar worker
# -----------------------------
docker rm -f worker_cpu || true

docker run -d \
  --restart unless-stopped \
  --name worker_cpu \
  -e COORDINADOR_HOST \
  -e COORDINADOR_PORT \
  -e RABBIT_HOST \
  -e RABBIT_USER \
  -e RABBIT_PASSWORD \
  -e RABBIT_PORT \
  -e RABBIT_VHOST \
  -e POOL_MANAGER_HOST \
  -e POOL_MANAGER_PORT \
  -e CPU_CAPACITY \
  -e HEARTBEAT_TTL \
  lucasrueda01/worker-cpu:dev2

echo "Worker CPU iniciado correctamente"

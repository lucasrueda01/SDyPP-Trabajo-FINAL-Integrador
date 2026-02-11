#!/bin/bash
set -e

echo "Inicializando worker CPU..."

# -----------------------------
# Variables de entorno
# -----------------------------
export COORDINADOR_HOST="${coordinator_host}"
export COORDINADOR_PORT="5000"

export RABBIT_HOST="${rabbit_host}"
export RABBIT_USER="blockchain"
export RABBIT_PASSWORD="blockchain123"
export RABBIT_PORT="5672"
export RABBIT_VHOST="blockchain"

export POOL_MANAGER_HOST="${pool_manager_host}"
export POOL_MANAGER_PORT="6000"

export CPU_CAPACITY="10"
export HEARTBEAT_TTL="180"
export HEARTBEAT_INTERVAL="10"
export HEARTBEAT_TIMEOUT="8"
export TZ="America/Argentina/Buenos_Aires"

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
  -e HEARTBEAT_INTERVAL \
  -e HEARTBEAT_TIMEOUT \
  -e TZ \
  ghcr.io/lucasrueda01/worker-cpu:latest

echo "Worker CPU iniciado correctamente"
#!/bin/bash
set -e

echo "Inicializando worker CPU..."

# =========================================
# 1️⃣ Función para leer metadata (si existe)
# =========================================
get_metadata() {
  curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

# =========================================
# 2️⃣ Obtener variables (metadata o fallback)
# =========================================

COORDINADOR_HOST=$(get_metadata "COORDINADOR_HOST")
RABBIT_HOST=$(get_metadata "RABBIT_HOST")
POOL_MANAGER_HOST=$(get_metadata "POOL_MANAGER_HOST")

if [ -z "$COORDINADOR_HOST" ]; then
  COORDINADOR_HOST="${coordinator_host}"
fi

if [ -z "$RABBIT_HOST" ]; then
  RABBIT_HOST="${rabbit_host}"
fi

if [ -z "$POOL_MANAGER_HOST" ]; then
  POOL_MANAGER_HOST="${pool_manager_host}"
fi


# Variables estáticas
COORDINADOR_PORT="5000"
RABBIT_USER="blockchain"
RABBIT_PASSWORD="blockchain123"
RABBIT_PORT="5672"
RABBIT_VHOST="blockchain"
POOL_MANAGER_PORT="6000"
CPU_CAPACITY="10"

HEARTBEAT_TTL="180"
HEARTBEAT_INTERVAL="10"
HEARTBEAT_TIMEOUT="8"
TZ="America/Argentina/Buenos_Aires"

export COORDINADOR_HOST
export COORDINADOR_PORT
export RABBIT_HOST
export RABBIT_USER
export RABBIT_PASSWORD
export RABBIT_PORT
export RABBIT_VHOST
export POOL_MANAGER_HOST
export POOL_MANAGER_PORT
export CPU_CAPACITY
export HEARTBEAT_TTL
export HEARTBEAT_INTERVAL
export HEARTBEAT_TIMEOUT
export TZ

echo "Variables configuradas correctamente"

# =========================================
# 3️⃣ Instalar Docker
# =========================================
apt-get update
apt-get install -y docker.io
systemctl enable --now docker

# =========================================
# 4️⃣ Ejecutar contenedor
# =========================================
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

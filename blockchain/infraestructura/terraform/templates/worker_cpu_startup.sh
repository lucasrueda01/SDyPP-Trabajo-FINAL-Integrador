#!/bin/bash
set -e

echo "======================================"
echo "Inicializando Worker CPU..."
echo "======================================"

echo "Esperando metadata..."

until curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/id > /dev/null; do
  echo "Metadata no disponible aún..."
  sleep 2
done

echo "Metadata disponible."


# =========================================
# 1️⃣ Resolver hosts (Terraform o Metadata)
# =========================================

# --- Si viene desde templatefile (workers base) ---
INGRESS_IP="${ingress_ip}"

if [ -n "$INGRESS_IP" ]; then
  echo "Modo: Worker Base (Terraform)"
  RABBIT_HOST="$INGRESS_IP"
  COORDINADOR_HOST="$INGRESS_IP"
  POOL_MANAGER_HOST="$INGRESS_IP"
else
  echo "Modo: Worker Dinámico (CPU Scaler)"
  
  RABBIT_HOST=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/RABBIT_HOST || true)

  COORDINADOR_HOST=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/COORDINADOR_HOST || true)

  POOL_MANAGER_HOST=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/POOL_MANAGER_HOST || true)
fi

echo "RABBIT_HOST=$RABBIT_HOST"
echo "COORDINADOR_HOST=$COORDINADOR_HOST"
echo "POOL_MANAGER_HOST=$POOL_MANAGER_HOST"

# Validación mínima
if [ -z "$RABBIT_HOST" ] || [ -z "$COORDINADOR_HOST" ] || [ -z "$POOL_MANAGER_HOST" ]; then
  echo "ERROR: No se pudieron resolver los hosts correctamente"
  exit 1
fi

# =========================================
# 2️⃣ Variables estáticas
# =========================================

COORDINADOR_PORT="80"
RABBIT_PORT="5672"
POOL_MANAGER_PORT="80"

RABBIT_USER="blockchain"
RABBIT_PASSWORD="blockchain123"
RABBIT_VHOST="blockchain"

CPU_CAPACITY="10"

HEARTBEAT_TTL="180"
HEARTBEAT_INTERVAL="10"
HEARTBEAT_TIMEOUT="8"

TZ="America/Argentina/Buenos_Aires"

# Exportar variables
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

apt-get update -y
echo "Instalando Docker..."
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

echo "======================================"
echo "Worker CPU iniciado correctamente"
echo "======================================"

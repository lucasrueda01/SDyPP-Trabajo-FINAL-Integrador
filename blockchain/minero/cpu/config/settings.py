import os
from dotenv import load_dotenv

load_dotenv()

# ======================
# RabbitMQ
# ======================

RABBIT_USER = os.getenv("RABBIT_USER")
RABBIT_PASSWORD = os.getenv("RABBIT_PASSWORD")
RABBIT_HOST = os.getenv("RABBIT_HOST")
RABBIT_PORT = os.getenv("RABBIT_PORT", "5672")
RABBIT_VHOST = os.getenv("RABBIT_VHOST", "")

RABBIT_URL = (
    f"amqp://{RABBIT_USER}:{RABBIT_PASSWORD}"
    f"@{RABBIT_HOST}:{RABBIT_PORT}/{RABBIT_VHOST}"
)
# ampqp://user:password@host:port/

# ======================
# Coordinador
# ======================
COORDINADOR_HOST = os.getenv("COORDINADOR_HOST", "localhost")
COORDINADOR_PORT = int(os.getenv("COORDINADOR_PORT", 5000))

DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

# ======================
# Pool Manager
# ======================
POOL_MANAGER_HOST = os.getenv("POOL_MANAGER_HOST", "localhost")
POOL_MANAGER_PORT = int(os.getenv("POOL_MANAGER_PORT", 6000))
CPU_CAPACITY = int(os.getenv("CPU_CAPACITY", 10))
HEARTBEAT_TTL = int(os.getenv("HEARTBEAT_TTL", 30))


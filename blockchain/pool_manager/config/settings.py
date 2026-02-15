import os
from dotenv import load_dotenv

load_dotenv()

# ======================
# Redis
# ======================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_TCP_PORT", 6379))

# ======================
# RabbitMQ
# ======================

RABBIT_USER = os.getenv("RABBIT_USER")
RABBIT_PASSWORD = os.getenv("RABBIT_PASSWORD")
RABBIT_HOST = os.getenv("RABBIT_HOST")
RABBIT_PORT = os.getenv("RABBIT_TCP_PORT", "5672")
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

# ======================
# Blockchain / Minería
# ======================
MAX_RANDOM = int(os.getenv("MAX_RANDOM", 99_999_999))

DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

# ======================
# Pool Manager
# ======================
COOPERATIVE_MINING = os.getenv("COOPERATIVE_MINING", "False").lower() in (
    "true",
    "1",
    "t",
)
POOL_MANAGER_HOST = os.getenv("POOL_MANAGER_HOST", "localhost")
POOL_MANAGER_PORT = int(os.getenv("POOL_MANAGER_PORT", 6000))
CPU_CAPACITY = int(os.getenv("CPU_CAPACITY", 10))
GPU_CAPACITY = int(os.getenv("GPU_CAPACITY", 100))
HEARTBEAT_TTL = int(os.getenv("HEARTBEAT_TTL", 90))
FRAGMENT_PERCENT = float(os.getenv("FRAGMENT_PERCENT", 0.5))

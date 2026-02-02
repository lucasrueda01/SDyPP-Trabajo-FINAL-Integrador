import os
from dotenv import load_dotenv

load_dotenv()

# ======================
# Redis
# ======================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ======================
# RabbitMQ
# ======================
RABBIT_HOST = os.getenv("RABBIT_HOST", "localhost")
RABBIT_USER = os.getenv("RABBIT_USER", "guest")
RABBIT_PASSWORD = os.getenv("RABBIT_PASSWORD", "guest")
RABBIT_URL = os.getenv("RABBIT_URL", "")    


# ======================
# Coordinador
# ======================
COORDINADOR_HOST = os.getenv("COORDINADOR_HOST", "localhost")
COORDINADOR_PORT = int(os.getenv("COORDINADOR_PORT", 5000))

# ======================
# Blockchain / Minería
# ======================
DIFFICULTY_LOW = int(os.getenv("DIFFICULTY_LOW", 3))
DIFFICULTY_HIGH = int(os.getenv("DIFFICULTY_HIGH", 6))
BASE_STRING_CHAIN = os.getenv("BASE_STRING_CHAIN", "A3F8")
CPUS_PER_GPU = int(os.getenv("CPUS_PER_GPU", 4))
MAX_RANDOM = int(os.getenv("MAX_RANDOM", 99_999_999))
MAX_TRANSACTIONS_PER_BLOCK = int(
    os.getenv("MAX_TRANSACTIONS_PER_BLOCK", 20)
)

# ======================
# Tiempos
# ======================
PROCESSING_TIME = int(os.getenv("PROCESSING_TIME", 15))

# ======================
# Storage / Credenciales
# ======================
BUCKET_NAME = os.getenv("BUCKET_NAME", "bucket_integrador2")

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
EXPECTED_GPUS = int(os.getenv("EXPECTED_GPUS", 1))
HEARTBEAT_TTL = int(os.getenv("HEARTBEAT_TTL", 10))
BASE_CPU_REPLICAS = int(os.getenv("BASE_CPU_REPLICAS", 2))
SCALE_COOLDOWN = int(os.getenv("SCALE_COOLDOWN", 60))  # segundos
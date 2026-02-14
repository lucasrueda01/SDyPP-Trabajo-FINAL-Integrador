import os
from dotenv import load_dotenv

load_dotenv()

# ======================
# Redis
# ======================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_TCP_PORT", 6379))

# ======================
# Blockchain / Minería
# ======================
CPUS_PER_GPU = int(os.getenv("CPUS_PER_GPU", 2))
EXPECTED_GPUS = int(os.getenv("EXPECTED_GPUS", 1))
BASE_CPU_REPLICAS = int(os.getenv("BASE_CPU_REPLICAS", 2))
SCALE_COOLDOWN = int(os.getenv("SCALE_COOLDOWN", 90))  # segundos
SCALE_INTERVAL = int(os.getenv("SCALE_INTERVAL", 30))  # segundos
AUTOSCALER = os.getenv("AUTOSCALER", "True").lower() in ("true", "1", "t")
HEARTBEAT_TTL = os.getenv("HEARTBEAT_TTL", 120)

DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
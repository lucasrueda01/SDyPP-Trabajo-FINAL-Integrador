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

QUEUE_NAME_TX = os.getenv("QUEUE_NAME_TX", "QueueTransactions")
EXCHANGE_BLOCK = os.getenv("EXCHANGE_BLOCK", "ExchangeBlock")

# ======================
# Coordinador
# ======================
COORDINADOR_HOST = os.getenv("COORDINADOR_HOST", "0.0.0.0")
COORDINADOR_PORT = int(os.getenv("COORDINADOR_PORT", 5000))

# ======================
# Blockchain / Minería
# ======================
PREFIX = os.getenv("PREFIX", "000")
BASE_STRING_CHAIN = os.getenv("BASE_STRING_CHAIN", "A3F8")

MAX_RANDOM = int(os.getenv("MAX_RANDOM", 99_999_999))
MAX_TRANSACTIONS_PER_BLOCK = int(
    os.getenv("MAX_TRANSACTIONS_PER_BLOCK", 20)
)

# ======================
# Tiempos
# ======================
TIMER = int(os.getenv("TIMER", 15))

# ======================
# Storage / Credenciales
# ======================
BUCKET_NAME = os.getenv("BUCKET_NAME", "bucket_integrador2")

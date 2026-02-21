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
COORDINADOR_PORT = int(os.getenv("COORDINADOR_PORT", 5000))

# ======================
# Blockchain / Minería
# ======================
DIFFICULTY_LOW = int(os.getenv("DIFFICULTY_LOW", 3))
DIFFICULTY_HIGH = int(os.getenv("DIFFICULTY_HIGH", 6))
BASE_STRING_CHAIN = os.getenv("BASE_STRING_CHAIN", "A3F8")
MAX_RANDOM = int(os.getenv("MAX_RANDOM", 99_999_999))
MAX_TRANSACTIONS_PER_BLOCK = int(os.getenv("MAX_TRANSACTIONS_PER_BLOCK", 100))
PROCESSING_TIME = int(os.getenv("PROCESSING_TIME", 15))

BUCKET_NAME = os.getenv("BUCKET_NAME", "bucket_integrador2")

FRAGMENT_PERCENT = float(os.getenv("FRAGMENT_PERCENT", 0.5))
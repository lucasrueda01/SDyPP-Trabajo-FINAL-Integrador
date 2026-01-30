import time
import json
import sys
import logging

from flask import Flask, request, jsonify
import redis
import pika

import config.settings as settings

app = Flask(__name__)

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = logging.DEBUG if getattr(settings, "DEBUG", False) else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("pool-manager")
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("redis").setLevel(logging.WARNING)

# -----------------------
# Cooldown state
# -----------------------
last_scale_time = 0

# -----------------------
# Redis
# -----------------------
def redisConnect():
    client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=True,
    )
    client.ping()
    logger.info("Conectado a Redis")
    return client


# -----------------------
# RabbitMQ
# -----------------------
def queueConnect():
    if settings.RABBIT_URL:
        params = pika.URLParameters(settings.RABBIT_URL)
    else:
        params = pika.ConnectionParameters(
            host=settings.RABBIT_HOST,
            credentials=pika.PlainCredentials(
                settings.RABBIT_USER, settings.RABBIT_PASSWORD
            ),
            heartbeat=600,
        )

    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.exchange_declare(
        exchange=settings.EXCHANGE_BLOCK,
        exchange_type="topic",
        durable=True,
    )

    channel.exchange_declare(
        exchange="blocks_competitive",
        exchange_type="fanout",
        durable=True,
    )

    logger.info("Conectado a RabbitMQ")
    return connection, channel


# -----------------------
# Workers (TTL-based)
# -----------------------
@app.route("/register", methods=["POST"])
def register_worker():
    data = request.get_json()
    wid = data["id"]

    worker_data = {
        "id": wid,
        "type": data["type"],
        "capacity": data.get("capacity", 5),
        "ip": request.remote_addr,
    }

    redisClient.set(
        f"worker:{wid}",
        json.dumps(worker_data),
        ex=settings.HEARTBEAT_TTL + 5, # Slightly more than TTL
    )

    logger.info(
        "Worker registrado: id=%s type=%s ip=%s",
        wid,
        data["type"],
        request.remote_addr,
    )

    return jsonify({"status": "ok"})


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    data = request.get_json()
    wid = data["id"]

    raw = redisClient.get(f"worker:{wid}")
    if not raw:
        logger.warning("Heartbeat de worker desconocido: %s", wid)
        return jsonify({"error": "unknown worker"}), 404

    # Renovar TTL
    redisClient.expire(f"worker:{wid}", settings.HEARTBEAT_TTL + 5)
    logger.debug("Heartbeat recibido de %s", wid)

    return jsonify({"status": "ok"})


def get_alive_workers():
    alive = []
    total_capacity = 0

    # Redis define quién está vivo
    for key in redisClient.scan_iter("worker:*"):
        raw = redisClient.get(key)
        if not raw:
            continue

        w = json.loads(raw)
        alive.append(w)
        total_capacity += int(w["capacity"])

    logger.info(
        "Workers vivos: %d | Capacidad total: %d",
        len(alive),
        total_capacity,
    )

    return alive, total_capacity


# -----------------------
# GPU -> CPU failover (conceptual)
# -----------------------
def handle_gpu_failover(alive_workers):
    global last_scale_time

    alive_gpus = [w for w in alive_workers if w["type"] == "gpu"]
    missing = settings.EXPECTED_GPUS - len(alive_gpus)

    if missing <= 0:
        return

    now = time.time()
    if now - last_scale_time < settings.SCALE_COOLDOWN:
        logger.info("Escalado omitido por cooldown")
        return

    cpus_per_gpu = settings.GPU_CAPACITY // settings.CPU_CAPACITY
    target_replicas = settings.BASE_CPU_REPLICAS + (missing * cpus_per_gpu)

    logger.warning(
        "[CONCEPTUAL] Faltan %d GPUs -> escalar CPUs a %d replicas",
        missing,
        target_replicas,
    )

    last_scale_time = now


# -----------------------
# Safe publish
# -----------------------
def safe_publish(exchange, routing_key, body):
    global connection, channel

    try:
        if connection.is_closed or channel.is_closed:
            connection, channel = queueConnect()

        channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )

    except pika.exceptions.AMQPError:
        logger.exception("Error publicando, reconectando")
        connection, channel = queueConnect()
        channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )


# -----------------------
# Dispatch
# -----------------------
@app.route("/dispatch", methods=["POST"])
def dispatch_block():
    block = request.get_json()
    block_id = block["blockId"]

    alive, total_capacity = get_alive_workers()
    handle_gpu_failover(alive)

    if not alive:
        logger.error("No hay workers vivos")
        return jsonify({"error": "no workers"}), 503

    # ---------- COMPETITIVO ----------
    if not settings.COOPERATIVE_MINING:
        logger.info("Despachando bloque %s en modo COMPETITIVO", block_id)

        safe_publish(
            exchange="blocks_competitive",
            routing_key="",
            body=json.dumps(block),
        )

        return jsonify({"mode": "competitive", "workers": len(alive)})

    # ---------- COOPERATIVO ----------
    logger.info("Despachando bloque %s en modo COOPERATIVO", block_id)

    nonce_start = 0
    nonce_end = block.get("numMaxRandom", settings.MAX_RANDOM)
    total_space = nonce_end - nonce_start + 1
    cursor = nonce_start

    assignments = []

    for w in alive:
        ratio = w["capacity"] / total_capacity
        window = max(1, int(total_space * ratio))

        start = cursor
        end = min(cursor + window - 1, nonce_end)

        payload = {
            **block,
            "nonce_start": start,
            "nonce_end": end,
        }

        logger.info("Asignando %s -> nonce %d-%d", w["id"], start, end)

        safe_publish(
            exchange=settings.EXCHANGE_BLOCK,
            routing_key="blocks",
            body=json.dumps(payload),
        )

        assignments.append(
            {"worker": w["id"], "nonce_start": start, "nonce_end": end}
        )

        cursor = end + 1

    return jsonify({"mode": "cooperative", "assignments": assignments})


# -----------------------
# Init
# -----------------------
logger.info("Pool Manager iniciando")
redisClient = redisConnect()
connection, channel = queueConnect()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.POOL_MANAGER_PORT)

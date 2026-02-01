import time
import json
import sys
import logging
import threading

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
        exchange="blocks_cooperative",
        exchange_type="topic",
        durable=True,
    )

    channel.exchange_declare(
        exchange="blocks_competitive",
        exchange_type="fanout",
        durable=True,
    )

    # Queue desde el coordinador
    channel.queue_declare(queue="pool_tasks", durable=True)
    # Dead Letter Exchange y Queue
    dlx_name = "dlx.tasks"
    gpu_ttl = 60000  # ms

    channel.exchange_declare(exchange=dlx_name, exchange_type="fanout", durable=True)
    channel.queue_declare(queue="queue.dlq", durable=True)
    channel.queue_bind(exchange=dlx_name, queue="queue.dlq")

    channel.queue_declare(
        queue="queue.gpu",
        durable=True,
        arguments={
            "x-message-ttl": gpu_ttl,
            "x-dead-letter-exchange": dlx_name,
        },
    )

    channel.queue_declare(queue="queue.cpu", durable=True)

    logger.info("Conectado a RabbitMQ (with DLQ support)")
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
        ex=settings.HEARTBEAT_TTL + 5,
    )  # Registro worker con expiracion (TTL)

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

    key = f"worker:{wid}"
    raw = redisClient.get(key)

    if not raw:
        logger.info("Worker %s no registrado, solicitando re-registro", wid)
        return jsonify({"error": "not registered"}), 404

    redisClient.expire(key, settings.HEARTBEAT_TTL + 5)
    logger.debug("Heartbeat recibido de %s", wid)

    return jsonify({"status": "ok"})



def get_alive_workers():
    alive = []
    total_capacity = 0

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
# GPU -> CPU failover
# -----------------------
# TODO
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

    cpus_per_gpu = settings.CPUS_PER_GPU
    target_replicas = settings.BASE_CPU_REPLICAS + (missing * cpus_per_gpu)

    logger.warning(
        "[CONCEPTUAL] Faltan %d GPUs -> escalar CPUs a %d replicas",
        missing,
        target_replicas,
    )

    last_scale_time = now


# -----------------------
# Fragmentacion de bloques
# -----------------------
def fragmentar(block, num_gpus, num_cpus):
    """Divide el espacio de búsqueda de nonce en fragmentos para GPU y CPU.
    - GPUs: la parte de GPU se divide en N fragmentos (uno por GPU).
    - CPUs: el espacio restante se divide en N fragmentos de CPU.
    Devuelve una tupla: (gpu_payloads, cpu_payloads)
    donde gpu_payloads es una lista (puede estar vacía)."""

    nonce_start = block.get("nonce_start", 0)
    nonce_end = block.get(
        "nonce_end",
        block.get("numMaxRandom", settings.MAX_RANDOM),
    )

    total_space = nonce_end - nonce_start + 1

    gpu_capacity_total = num_gpus * settings.GPU_CAPACITY
    cpu_capacity_total = num_cpus * settings.CPU_CAPACITY
    total_capacity = gpu_capacity_total + cpu_capacity_total

    if total_capacity == 0:
        return [], []

    gpu_space = 0
    if gpu_capacity_total > 0:
        gpu_space = int(total_space * (gpu_capacity_total / total_capacity))

    cursor = nonce_start

    # ---- GPUs: split into N chunks ----
    gpu_payloads = []
    if num_gpus > 0 and gpu_space > 0:
        gpu_chunk = gpu_space // num_gpus
        gpu_cursor = cursor

        for i in range(num_gpus):
            start = gpu_cursor
            if i == num_gpus - 1:
                end = cursor + gpu_space - 1
            else:
                end = gpu_cursor + gpu_chunk - 1

            payload = {**block, "nonce_start": start, "nonce_end": end}
            gpu_payloads.append(payload)
            gpu_cursor = end + 1

        cursor = cursor + gpu_space

    # ---- CPUs: split remaining space ----
    cpu_payloads = []
    if num_cpus > 0 and cursor <= nonce_end:
        cpu_start = cursor
        cpu_end = nonce_end
        cpu_total_space = cpu_end - cpu_start + 1

        cpu_chunk = cpu_total_space // num_cpus
        cpu_cursor = cpu_start

        for i in range(num_cpus):
            start = cpu_cursor
            if i == num_cpus - 1:
                end = cpu_end
            else:
                end = cpu_cursor + cpu_chunk - 1

            payload = {**block, "nonce_start": start, "nonce_end": end}
            cpu_payloads.append(payload)
            cpu_cursor = end + 1

    return gpu_payloads, cpu_payloads


# -----------------------
# Safe publish que verifica conexion
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

def dispatch_to_workers(block):
    block_id = block["blockId"]

    alive, _ = get_alive_workers()
    handle_gpu_failover(alive)

    if not alive:
        logger.warning("No hay workers vivos para %s. Reintentando...", block_id)
        return False

    # ---------- COMPETITIVO ----------
    if not settings.COOPERATIVE_MINING:
        logger.info("Despachando %s en COMPETITIVO", block_id)

        safe_publish(
            exchange="blocks_competitive",
            routing_key="",
            body=json.dumps(block),
        )
        return True

    # ---------- COOPERATIVO ----------
    logger.info("Despachando %s en COOPERATIVO", block_id)

    gpu_workers = [w for w in alive if w["type"] == "gpu"]
    cpu_workers = [w for w in alive if w["type"] == "cpu"]

    num_gpus = len(gpu_workers)
    num_cpus = len(cpu_workers)

    gpu_payloads, cpu_payloads = fragmentar(block, num_gpus, num_cpus)

    for i, payload in enumerate(gpu_payloads):
        start = payload["nonce_start"]
        end = payload["nonce_end"]

        logger.info(
            "Asignando GPU %d/%d -> nonce %d-%d",
            i + 1,
            len(gpu_payloads),
            start,
            end,
        )

        safe_publish(
            exchange="blocks_cooperative",
            routing_key="blocks.gpu",
            body=json.dumps(payload),
        )

    for i, payload in enumerate(cpu_payloads):
        start = payload["nonce_start"]
        end = payload["nonce_end"]
        logger.info(
            "Asignando CPU %d/%d -> nonce %d-%d",
            i + 1,
            len(cpu_payloads),
            start,
            end,
        )

        safe_publish(
            exchange="blocks_cooperative",
            routing_key="blocks.cpu",
            body=json.dumps(payload),
        )

    return True


# -----------------------
# Consumidor DLQ: inspecciona mensajes que expiraron de queue.gpu
# y decide si republicar a GPUs o fragmentar para CPUs.
# -----------------------
def start_dlq_consumer():
    conn, ch = queueConnect()
    ch.basic_qos(prefetch_count=1)

    def on_dlq_message(channel, method, properties, body):
        try:
            msg = json.loads(body)
            logger.warning("DLQ: recibido mensaje expirado: %s", msg.get("blockId"))

            alive, _ = get_alive_workers()
            num_gpus = len([w for w in alive if w["type"] == "gpu"])
            num_cpus = len([w for w in alive if w["type"] == "cpu"])

            # Si hay GPUs vivas, intentar republicar a queue.gpu
            if num_gpus > 0:
                logger.info("DLQ: hay GPUs vivas -> republicando a queue.gpu")

                gpu_payload, cpu_payloads = fragmentar(msg, num_gpus, num_cpus)
                if gpu_payload:
                    safe_publish(
                        exchange="blocks_cooperative",
                        routing_key="blocks.gpu",
                        body=json.dumps(gpu_payload),
                    )
                for payload in cpu_payloads:
                    safe_publish(
                        exchange="blocks_cooperative",
                        routing_key="blocks.cpu",
                        body=json.dumps(payload),
                    )

            else:
                # No hay GPUs vivas, fragmentar para CPUs
                logger.info(
                    "DLQ: no hay GPUs -> fragmentando para CPUs y publicando a queue.cpu"
                )

                _, cpu_payloads = fragmentar(msg, 0, num_cpus)
                for payload in cpu_payloads:
                    # Podriamos disminuir el rango nonce, pero eso lo hace el coordinador
                    safe_publish(
                        exchange="blocks_cooperative",
                        routing_key="blocks.cpu",
                        body=json.dumps(payload),
                    )

            channel.basic_ack(delivery_tag=method.delivery_tag)

        except Exception:
            logger.exception("Error procesando DLQ message, requeueing")
            try:
                channel.basic_nack(method.delivery_tag, requeue=True)
            except Exception:
                pass

    ch.basic_consume(
        queue="queue.dlq",
        on_message_callback=on_dlq_message,
        auto_ack=False,
    )

    logger.info("DLQ consumer started")
    ch.start_consuming()


def start_pool_consumer():
    conn, ch = queueConnect()
    ch.basic_qos(prefetch_count=1)

    def on_pool_task(channel, method, properties, body):
        try:
            block = json.loads(body)
            logger.info("Recibido bloque %s desde pool_tasks", block["blockId"])

            ok = dispatch_to_workers(block)

            if ok:
                channel.basic_ack(method.delivery_tag)
            else:
                time.sleep(10)
                channel.basic_nack(method.delivery_tag, requeue=True)

        except Exception:
            logger.exception("Error procesando pool_task")
            channel.basic_nack(method.delivery_tag, requeue=True)

    ch.basic_consume(
        queue="pool_tasks",
        on_message_callback=on_pool_task,
        auto_ack=False,
    )

    logger.info("Pool Manager consumiendo pool_tasks")
    ch.start_consuming()


# -----------------------
# Init
# -----------------------
logger.info("Pool Manager iniciando")

redisClient = redisConnect()
connection, channel = queueConnect()

consumer_thread = threading.Thread(
    target=start_pool_consumer,
    daemon=True,
)
consumer_thread.start()

dlq_thread = threading.Thread(target=start_dlq_consumer, daemon=True)
dlq_thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.POOL_MANAGER_PORT)

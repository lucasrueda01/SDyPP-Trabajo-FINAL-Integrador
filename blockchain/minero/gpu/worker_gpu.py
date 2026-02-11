# worker_gpu.py
import threading
import time
import json
import random
from threading import Thread
import requests
import logging
import sys

import pika
from minero_gpu import ejecutar_minero
import config.settings as settings

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = logging.DEBUG if getattr(settings, "DEBUG", False) else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("worker_gpu")
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# -----------------------
WORKER_ID = f"gpu-{random.randint(1000,9999)}"

EXCHANGE_COMPETITIVE = "blocks_competitive"
EXCHANGE_COOPERATIVE = "blocks_cooperative"

hostCoordinador = settings.COORDINADOR_HOST
puertoCoordinador = settings.COORDINADOR_PORT

rabbit_url = settings.RABBIT_URL
hostRabbit = settings.RABBIT_HOST
rabbitUser = settings.RABBIT_USER
rabbitPassword = settings.RABBIT_PASSWORD

pool_manager_host = settings.POOL_MANAGER_HOST
pool_manager_port = settings.POOL_MANAGER_PORT


# -----------------------
# Utils
# -----------------------
def enviar_resultado(data: dict, retries: int = 2, timeout: int = 5) -> int | None:
    url = f"http://{hostCoordinador}:{puertoCoordinador}/solved_task"
    backoff = 1
    for i in range(retries + 1):
        try:
            resp = requests.post(url, json=data, timeout=timeout)
            logger.debug("POST %s -> %s", url, resp.status_code)
            if getattr(settings, "DEBUG", False):
                logger.debug("Coordinator response: %s", resp.text)
            return resp.status_code
        except requests.exceptions.RequestException as e:
            logger.warning(
                "[%s] Intento %d: fallo POST al coordinador: %s",
                WORKER_ID,
                i + 1,
                e,
            )
            if i < retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error("[%s] No se pudo enviar resultado", WORKER_ID)
                return None


# -----------------------
# Consumidor
# -----------------------
def on_message_received(ch, method, _, body):
    try:
        try:
            data = json.loads(body)
        except Exception:
            logger.exception("[%s] Error decodificando JSON", WORKER_ID)
            ch.basic_ack(method.delivery_tag)
            return

        required = (
            "blockId",
            "numMaxRandom",
            "prefijo",
            "baseStringChain",
            "blockchainContent",
        )
        if not all(k in data for k in required):
            logger.error("[%s] Bloque incompleto: %s", WORKER_ID, data)
            ch.basic_ack(method.delivery_tag)
            return

        block_id = data["blockId"]
        prefix = data["prefijo"]
        hash_base = data["baseStringChain"] + data["blockchainContent"]

        if "nonce_start" in data and "nonce_end" in data:
            from_nonce = int(data["nonce_start"])
            to_nonce = int(data["nonce_end"])
            mode = "COOPERATIVO"
        else:
            from_nonce = 1
            to_nonce = int(data["numMaxRandom"])
            mode = "COMPETITIVO"

        logger.info(
            "[%s] Bloque %s | modo=%s | rango=[%d-%d]",
            WORKER_ID,
            block_id,
            mode,
            from_nonce,
            to_nonce,
        )

        start_time = time.time()
        resultado_raw = ejecutar_minero(from_nonce, to_nonce, prefix, hash_base)
        processing_time = time.time() - start_time

        try:
            resultado = json.loads(resultado_raw)
        except Exception:
            logger.exception("[%s] Respuesta inválida del minero GPU", WORKER_ID)
            ch.basic_ack(method.delivery_tag)
            return

        if not resultado.get("hash_md5_result"):
            logger.info(
                "[%s] No se encontró solución | time=%.2fs",
                WORKER_ID,
                processing_time,
            )
            enviar_resultado(
                {
                    "blockId": block_id,
                    "workerId": WORKER_ID,
                    "type": "gpu",
                    "processingTime": processing_time,
                    "hashRate": 0.0,
                    "hash": "",
                    "result": "",
                }
            )
            ch.basic_ack(method.delivery_tag)
            return

        intentos = int(resultado.get("intentos", to_nonce - from_nonce))
        hash_rate = intentos / processing_time if processing_time > 0 else 0

        logger.info(
            "[%s] Solución encontrada | nonce=%s | time=%.2fs | H/s=%.2f",
            WORKER_ID,
            resultado["numero"],
            processing_time,
            hash_rate,
        )

        status = enviar_resultado(
            {
                "blockId": block_id,
                "workerId": WORKER_ID,
                "type": "gpu",
                "processingTime": processing_time,
                "hashRate": hash_rate,
                "hash": resultado["hash_md5_result"],
                "result": str(resultado["numero"]),
            }
        )

        if status == 201:
            logger.info("[%s] Bloque aceptado por el coordinador", WORKER_ID)
        elif status is None:
            logger.error("[%s] No se pudo comunicar con coordinador", WORKER_ID)
        else:
            logger.info(
                "[%s] Resultado descartado por el coordinador (status=%s)",
                WORKER_ID,
                status,
            )

        ch.basic_ack(method.delivery_tag)

    except Exception:
        logger.exception("[%s] Error procesando mensaje", WORKER_ID)


# -----------------------
# Heartbeat
# -----------------------
def heartbeat_loop():
    url = f"http://{pool_manager_host}:{pool_manager_port}/heartbeat"
    payload = {
        "id": WORKER_ID,
        "type": "gpu",
        "capacity": settings.GPU_CAPACITY,
    }

    logger.info("[%s] Heartbeat thread iniciado", WORKER_ID)

    while True:
        try:
            requests.post(url, json=payload, timeout=settings.HEARTBEAT_TIMEOUT)
        except Exception as e:
            logger.warning("[%s] No se pudo enviar heartbeat", WORKER_ID)
            logger.error("[%s]", e)

        time.sleep(settings.HEARTBEAT_INTERVAL)


# -----------------------
# RabbitMQ
# -----------------------
def connect_rabbit():
    while True:
        try:
            logger.info("[%s] Conectando a RabbitMQ...", WORKER_ID)
            params = (
                pika.URLParameters(rabbit_url)
                if rabbit_url
                else pika.ConnectionParameters(
                    host=hostRabbit,
                    credentials=pika.PlainCredentials(rabbitUser, rabbitPassword),
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )
            )
            connection = pika.BlockingConnection(params)
            logger.info("[%s] Conectado a RabbitMQ", WORKER_ID)
            return connection
        except pika.exceptions.AMQPConnectionError:
            logger.warning(
                "[%s] RabbitMQ no disponible, reintentando en 5s...", WORKER_ID
            )
            time.sleep(5)


# -----------------------
# Main
# -----------------------
def main():
    connection = connect_rabbit()
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)

    # -------- COMPETITIVO --------
    channel.exchange_declare(
        exchange=EXCHANGE_COMPETITIVE,
        exchange_type="fanout",
        durable=True,
    )

    result = channel.queue_declare("", exclusive=True)
    queue_competitive = result.method.queue
    channel.queue_bind(exchange=EXCHANGE_COMPETITIVE, queue=queue_competitive)

    # -------- COOPERATIVO --------
    channel.exchange_declare(
        exchange=EXCHANGE_COOPERATIVE,
        exchange_type="topic",
        durable=True,
    )

    # DLX (idempotente)
    channel.exchange_declare(
        exchange="dlx.tasks",
        exchange_type="fanout",
        durable=True,
    )

    # Binding DLQ
    channel.queue_bind(
        exchange="dlx.tasks",
        queue="queue.dlq",
        routing_key="dlq",
    )

    channel.queue_bind(
        exchange=EXCHANGE_COOPERATIVE,
        queue="queue.gpu",
        routing_key="blocks.gpu",
    )

    # Consumimos de ambos
    channel.basic_consume(
        queue=queue_competitive,
        on_message_callback=on_message_received,
        auto_ack=False,
    )

    channel.basic_consume(
        queue="queue.gpu",
        on_message_callback=on_message_received,
        auto_ack=False,
    )

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    logger.info("[%s] Worker GPU listo y esperando bloques...", WORKER_ID)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("[%s] Worker detenido por usuario", WORKER_ID)
        try:
            connection.close()
        except Exception:
            logger.exception("[%s] Error cerrando conexion RabbitMQ", WORKER_ID)


if __name__ == "__main__":
    main()

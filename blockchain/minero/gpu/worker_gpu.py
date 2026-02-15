import json
import random
import time
import logging
import sys
from threading import Thread

import requests
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
# Constantes
# -----------------------
EXCHANGE_COMPETITIVE = "blocks_competitive"
EXCHANGE_COOPERATIVE = "blocks_cooperative"

hostCoordinador = "coordinator." + settings.COORDINADOR_HOST + ".nip.io"
hostRabbit = settings.RABBIT_HOST
rabbitUser = settings.RABBIT_USER
rabbitPassword = settings.RABBIT_PASSWORD

pool_manager_host = "pool." + settings.POOL_MANAGER_HOST + ".nip.io"

logger.info(
    "Configuración del worker GPU: coordinador=%s, rabbit=%s, pool_manager=%s",
    hostCoordinador,
    hostRabbit,
    pool_manager_host,
)


# -----------------------
# Envío resultado
# -----------------------
def enviar_resultado(data: dict, retries: int = 2, timeout: int = 5) -> int | None:
    url = f"http://{hostCoordinador}/solved_task"
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
                "[%s] Intento %d: fallo al enviar resultado: %s",
                WORKER_ID,
                i + 1,
                e,
            )

            if i < retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(
                    "[%s] No se pudo enviar resultado tras %d intentos",
                    WORKER_ID,
                    retries + 1,
                )
                logger.debug("Payload que falló: %s", data)
                return None


# -----------------------
# RabbitMQ
# -----------------------
def connect_rabbit():
    while True:
        try:
            logger.info("[%s] Conectando a RabbitMQ...", WORKER_ID)

            credentials = pika.PlainCredentials(
                rabbitUser,
                rabbitPassword,
            )

            params = pika.ConnectionParameters(
                host=hostRabbit,
                port=int(settings.RABBIT_PORT),
                virtual_host=settings.RABBIT_VHOST or "/",
                credentials=credentials,
                heartbeat=20,
                blocked_connection_timeout=120,
                connection_attempts=10,
                retry_delay=5,
                socket_timeout=10,
            )

            connection = pika.BlockingConnection(params)

            logger.info(
                "[%s] Conectado correctamente a RabbitMQ con %s:%s",
                WORKER_ID,
                params.host,
                params.port,
            )

            logger.debug("[%s] Parámetros de conexión: %s", WORKER_ID, params)

            return connection

        except pika.exceptions.AMQPConnectionError:
            logger.warning(
                "[%s] RabbitMQ no disponible, reintentando en 5s...",
                WORKER_ID,
            )
            time.sleep(5)

        except Exception:
            logger.exception(
                "[%s] Error inesperado conectando a RabbitMQ",
                WORKER_ID,
            )
            time.sleep(5)


def declare_queues(channel):
    # -------- COMPETITIVO --------
    channel.exchange_declare(
        exchange=EXCHANGE_COMPETITIVE,
        exchange_type="fanout",
        durable=True,
    )

    result = channel.queue_declare(
        "", exclusive=True, arguments={"x-queue-type": "classic"}
    )
    queue_competitive = result.method.queue

    channel.queue_bind(
        exchange=EXCHANGE_COMPETITIVE,
        queue=queue_competitive,
    )

    # -------- COOPERATIVO --------
    channel.exchange_declare(
        exchange=EXCHANGE_COOPERATIVE,
        exchange_type="topic",
        durable=True,
    )

    channel.queue_declare(
        queue="queue.workers",
        durable=True,
        arguments={"x-queue-type": "quorum"},
    )

    channel.queue_bind(
        exchange=EXCHANGE_COOPERATIVE,
        queue="queue.workers",
    )

    # -------- Consumers --------
    channel.basic_consume(
        queue=queue_competitive,
        on_message_callback=on_message_received,
        auto_ack=False,
    )

    channel.basic_consume(
        queue="queue.workers",
        on_message_callback=on_message_received,
        auto_ack=False,
    )


# -----------------------
# Consumidor
# -----------------------
def on_message_received(channel, method, _, body):
    ack = False

    try:
        try:
            data = json.loads(body)
        except Exception:
            logger.exception("[%s] Error decodificando JSON", WORKER_ID)
            channel.basic_ack(delivery_tag=method.delivery_tag)
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
            channel.basic_ack(delivery_tag=method.delivery_tag)
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
            "[%s] Bloque %s | modo=%s | rango=[%d-%d] | prefijo=%s",
            WORKER_ID,
            block_id,
            mode,
            from_nonce,
            to_nonce,
            prefix,
        )

        start_time = time.time()
        resultado_raw = ejecutar_minero(from_nonce, to_nonce, prefix, hash_base)
        processing_time = time.time() - start_time

        try:
            resultado = json.loads(resultado_raw)
        except Exception:
            logger.exception("[%s] Respuesta inválida del minero GPU", WORKER_ID)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        ack = True

        if not resultado.get("hash_md5_result"):
            logger.info(
                "[%s] No se encontró solución | block=%s | time=%.2fs",
                WORKER_ID,
                block_id,
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
        else:
            intentos = int(resultado.get("intentos", to_nonce - from_nonce))
            hash_rate = intentos / processing_time if processing_time > 0 else 0

            logger.info(
                "[%s] Solución encontrada | block=%s | nonce=%s | time=%.2fs | H/s=%.2f",
                WORKER_ID,
                block_id,
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
                    "[%s] Resultado descartado (status=%s)",
                    WORKER_ID,
                    status,
                )

    except Exception:
        logger.exception("[%s] Error procesando mensaje", WORKER_ID)

    finally:
        try:
            if ack:
                channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            logger.exception("[%s] Error al ackear mensaje", WORKER_ID)


# -----------------------
# Heartbeat
# -----------------------
def heartbeat_loop():
    url = f"http://{pool_manager_host}/heartbeat"

    payload = {
        "id": WORKER_ID,
        "type": "gpu",
        "capacity": settings.GPU_CAPACITY,
    }

    while True:
        try:
            requests.post(url, json=payload, timeout=settings.HEARTBEAT_TIMEOUT)
        except Exception as e:
            logger.warning("[%s] No se pudo enviar heartbeat", WORKER_ID)
            logger.error("[%s]", e)

        time.sleep(settings.HEARTBEAT_INTERVAL)


# -----------------------
# Main
# -----------------------
def main():
    global WORKER_ID
    WORKER_ID = f"gpu-{random.randint(1000, 9999)}"

    while True:
        try:
            logger.info("[%s] Iniciando worker...", WORKER_ID)

            connection = connect_rabbit()
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)

            declare_queues(channel)
            hb_thread = Thread(
                target=heartbeat_loop,
                daemon=True,
                name="heartbeat-thread",
            )
            hb_thread.start()

            logger.info(
                "[%s] Worker GPU listo y esperando bloques...",
                WORKER_ID,
            )

            channel.start_consuming()

        except KeyboardInterrupt:
            logger.info("[%s] Worker detenido por usuario", WORKER_ID)
            try:
                connection.close()
            except Exception:
                pass
            break

        except Exception:
            logger.exception(
                "[%s] Conexión perdida o error inesperado. Reconectando en 5s...",
                WORKER_ID,
            )
            try:
                connection.close()
            except Exception:
                pass

            time.sleep(5)


if __name__ == "__main__":
    main()

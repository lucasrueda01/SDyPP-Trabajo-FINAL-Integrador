# worker_gpu.py
import time
import json
import random
import requests
import logging
import sys

import pika
from minero.gpu import minero_gpu
import config.settings as settings

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = logging.DEBUG if getattr(settings, "DEBUG", False) else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("worker_gpu")

logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# -----------------------
# Configuración
# -----------------------
hostRabbit = settings.RABBIT_HOST
exchangeBlock = "blocks_cooperative"  # topic
hostCoordinador = settings.COORDINADOR_HOST
puertoCoordinador = settings.COORDINADOR_PORT
rabbitUser = settings.RABBIT_USER
rabbitPassword = settings.RABBIT_PASSWORD
rabbit_url = settings.RABBIT_URL

WORKER_ID = f"gpu-{random.randint(1000,9999)}"


# -----------------------
# Envío de resultado al coordinador
# -----------------------
def enviar_resultado(data: dict, retries: int = 2) -> int | None:
    url = f"http://{hostCoordinador}:{puertoCoordinador}/solved_task"
    backoff = 1
    for i in range(retries + 1):
        try:
            response = requests.post(url, json=data, timeout=5)
            logger.debug("POST %s -> %s", url, response.status_code)
            return response.status_code
        except requests.exceptions.RequestException as e:
            logger.warning(
                "[%s] Fallo POST al coordinador (intento %d): %s", WORKER_ID, i + 1, e
            )
            if i < retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(
                    "[%s] No se pudo enviar resultado al coordinador", WORKER_ID
                )
                return None


def on_message_received(ch, method, _, body):
    try:
        try:
            data = json.loads(body)
        except Exception:
            logger.exception("[%s] Error decodificando JSON recibido", WORKER_ID)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        required = (
            "blockId",
            "numMaxRandom",
            "prefijo",
            "baseStringChain",
            "blockchainContent",
        )
        if not all(k in data for k in required):
            logger.error(
                "[%s] Bloque recibido incompleto: %s",
                WORKER_ID,
                {k: data.get(k) for k in data},
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        logger.info(
            "[%s] Bloque recibido: %s (prefijo=%s) (numMaxRandom=%s)",
            WORKER_ID,
            data["blockId"],
            data["prefijo"],
            data["numMaxRandom"],
        )

        from_val = 1
        to_val = int(data["numMaxRandom"])
        prefix = data["prefijo"]
        hash_base = data["baseStringChain"] + data["blockchainContent"]

        startTime = time.time()
        logger.info("[%s] Iniciando minero GPU", WORKER_ID)

        resultado_raw = minero_gpu.ejecutar_minero(from_val, to_val, prefix, hash_base)

        try:
            resultado = json.loads(resultado_raw)
        except Exception:
            logger.exception("[%s] Respuesta inválida del minero GPU", WORKER_ID)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        processingTime = time.time() - startTime

        # GPU no encontró solución
        if not resultado.get("hash_md5_result") or resultado.get("numero", 0) == 0:
            logger.info(
                "[%s] GPU no encontró solución en el rango (%.2fs)",
                WORKER_ID,
                processingTime,
            )
            dataResult = {
                "blockId": data["blockId"],
                "workerId": WORKER_ID,
                "processingTime": processingTime,
                "hashRate": 0,
                "hash": "",
                "result": "",
            }
            enviar_resultado(dataResult)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        intentos = int(resultado.get("intentos", to_val))
        hash_rate = intentos / processingTime if processingTime > 0 else 0

        dataResult = {
            "blockId": data["blockId"],
            "workerId": WORKER_ID,
            "processingTime": processingTime,
            "hashRate": hash_rate,
            "hash": resultado["hash_md5_result"],
            "result": str(resultado["numero"]),
        }

        logger.info(
            "[%s] Solución encontrada (nonce=%s | %.2fs | H/s=%.2f)",
            WORKER_ID,
            resultado["numero"],
            processingTime,
            hash_rate,
        )

        status = enviar_resultado(dataResult)
        if status == 201:
            logger.info("[%s] Bloque aceptado por el coordinador", WORKER_ID)
        elif status is None:
            logger.error("[%s] No se pudo contactar al coordinador", WORKER_ID)
        else:
            logger.info("[%s] Resultado descartado por el coordinador", WORKER_ID)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception:
        logger.exception("[%s] Error inesperado en worker GPU", WORKER_ID)
        ch.basic_ack(delivery_tag=method.delivery_tag)

# -----------------------
# Rabbit connection
# -----------------------
def connect_rabbit():
    while True:
        try:
            logger.info("[%s] Conectando a RabbitMQ...", WORKER_ID)
            if rabbit_url:
                params = pika.URLParameters(rabbit_url)
            else:
                params = pika.ConnectionParameters(
                    host=hostRabbit,
                    credentials=pika.PlainCredentials(rabbitUser, rabbitPassword),
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )

            connection = pika.BlockingConnection(params)
            logger.info("[%s] Conectado a RabbitMQ", WORKER_ID)
            return connection
        except pika.exceptions.AMQPConnectionError:
            logger.warning(
                "[%s] RabbitMQ no disponible, reintentando en 5s...", WORKER_ID
            )
            time.sleep(5)
        except Exception:
            logger.exception("[%s] Error inesperado conectando a RabbitMQ", WORKER_ID)
            time.sleep(5)

# -----------------------
# Main
# -----------------------
def main():
    try:
        logger.info("[%s] Conectando a RabbitMQ...", WORKER_ID)

        connection = connect_rabbit()
        channel = connection.channel()
        channel.exchange_declare(
            exchange=exchangeBlock,
            exchange_type="topic",
            durable=True,
        )

        result = channel.queue_declare("", exclusive=True)
        queue_name = result.method.queue
        channel.queue_bind(
            exchange=exchangeBlock,
            queue=queue_name,
            routing_key="blocks",
        )

        channel.basic_consume(
            queue=queue_name,
            on_message_callback=on_message_received,
            auto_ack=False,
        )

        logger.info("[%s] Worker GPU listo y esperando bloques...", WORKER_ID)
        channel.start_consuming()

    except KeyboardInterrupt:
        logger.info("[%s] Worker GPU detenido por usuario", WORKER_ID)
    except Exception:
        logger.exception("[%s] Error fatal en worker GPU", WORKER_ID)
    finally:
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

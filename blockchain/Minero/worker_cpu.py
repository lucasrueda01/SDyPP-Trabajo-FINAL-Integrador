import json
import hashlib
import random
import requests
import time
import sys
import logging

import pika
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
logger = logging.getLogger("worker_cpu")

logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# -----------------------
# Configuracion
# -----------------------
hostRabbit = settings.RABBIT_HOST
exchangeBlock = settings.EXCHANGE_BLOCK
hostCoordinador = settings.COORDINADOR_HOST
puertoCoordinador = settings.COORDINADOR_PORT
rabbitUser = settings.RABBIT_USER
rabbitPassword = settings.RABBIT_PASSWORD
rabbit_url = settings.RABBIT_URL

WORKER_ID = f"cpu-{random.randint(1000, 9999)}"


# -----------------------
# Helpers
# -----------------------
def calculateHash(data: str) -> str:
    hash_md5 = hashlib.md5()
    hash_md5.update(data.encode("utf-8"))
    return hash_md5.hexdigest()


def enviar_resultado(data: dict, retries: int = 2, timeout: int = 5) -> int | None:
    """
    Envía resultado al coordinador. Reintenta en caso de errores transitorios.
    Devuelve status code (int) o None si no pudo comunicarse.
    """
    url = f"http://{hostCoordinador}:{puertoCoordinador}/solved_task"
    backoff = 1
    for i in range(retries + 1):
        try:
            resp = requests.post(url, json=data, timeout=timeout)
            logger.debug("POST %s -> %s", url, resp.status_code)
            # log minimal del body para debug si está en DEBUG
            if getattr(settings, "DEBUG", False):
                logger.debug("Coordinator response: %s", resp.text)
            return resp.status_code
        except requests.exceptions.RequestException as e:
            logger.warning(
                "Intento %d: fallo al enviar resultado al coordinador: %s", i + 1, e
            )
            if i < retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(
                    "No se pudo enviar resultado al coordinador tras %d intentos",
                    retries + 1,
                )
                logger.debug("Payload que falló: %s", data)
                return None


# -----------------------
# Rabbit connection
# -----------------------
def connect_rabbit():
    while True:
        try:
            logger.info("[%s] Conectando a RabbitMQ...", WORKER_ID)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=hostRabbit,
                    credentials=pika.PlainCredentials(rabbitUser, rabbitPassword),
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )
            )
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


def ejecutar_minero(from_val: int, to_val: int, prefijo: str, hash_base: str):
    """
    Minado CPU secuencial.

    Args:
        from_val: nonce inicial (inclusive)
        to_val: nonce final (inclusive)
        prefijo: string de prefijo requerido (ej. '0000')
        hash_base: base string (baseStringChain + blockchainContent)

    Returns:
        dict con:
            - encontrado (bool)
            - numero (str)  -> nonce
            - hash_md5_result (str)
            - intentos (int)
            - processingTime (float)
            - hashRate (float)
    """
    
    start_time = time.time()
    intentos = 0

    for nonce in range(from_val, to_val + 1):
        intentos += 1
        nonce_str = str(nonce)

        hash_calculado = calculateHash(nonce_str + hash_base)

        if hash_calculado.startswith(prefijo):
            processing_time = time.time() - start_time
            hash_rate = intentos / processing_time if processing_time > 0 else 0.0

            return {
                "encontrado": True,
                "numero": nonce_str,
                "hash_md5_result": hash_calculado,
                "intentos": intentos,
                "processingTime": processing_time,
                "hashRate": hash_rate,
            }

    # No encontrado
    processing_time = time.time() - start_time
    return {
        "encontrado": False,
        "numero": "",
        "hash_md5_result": "",
        "intentos": intentos,
        "processingTime": processing_time,
        "hashRate": 0.0,
    }


# -----------------------
# Message handler
# -----------------------
def on_message_received(channel, method, _, body):
    try:
        # -----------------------
        # Decodificación del mensaje
        # -----------------------
        try:
            data = json.loads(body)
        except Exception:
            logger.exception("[%s] Error decodificando mensaje JSON", WORKER_ID)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        required_fields = (
            "blockId",
            "numMaxRandom",
            "baseStringChain",
            "blockchainContent",
            "prefijo",
        )
        if not all(k in data for k in required_fields):
            logger.error(
                "[%s] Mensaje de bloque incompleto. Data: %s",
                WORKER_ID,
                data,
            )
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        block_id = data["blockId"]
        max_nonce = int(data["numMaxRandom"])
        prefijo = data["prefijo"]

        # Base de hash: EXACTAMENTE igual que GPU
        hash_base = data["baseStringChain"] + data["blockchainContent"]

        logger.info(
            "[%s] Bloque recibido: %s (prefijo=%s, max=%s)",
            WORKER_ID,
            block_id,
            prefijo,
            max_nonce,
        )

        # -----------------------
        # Minado CPU (función separada)
        # -----------------------

        resultado = ejecutar_minero(
            1,
            max_nonce,
            prefijo,
            hash_base,
        )

        # -----------------------
        # Resultado
        # -----------------------
        if resultado["encontrado"]:
            logger.info(
                "[%s] Solución encontrada | block=%s | nonce=%s | time=%.2fs | attempts=%d | H/s=%.2f",
                WORKER_ID,
                block_id,
                resultado["numero"],
                resultado["processingTime"],
                resultado["intentos"],
                resultado["hashRate"],
            )

            dataResult = {
                "blockId": block_id,
                "workerId": WORKER_ID,
                "processingTime": resultado["processingTime"],
                "hashRate": resultado["hashRate"],
                "hash": resultado["hash_md5_result"],
                "result": resultado["numero"],
            }

            status = enviar_resultado(dataResult)
            if status == 201:
                logger.info("[%s] Bloque aceptado por el coordinador", WORKER_ID)
            elif status is None:
                logger.error(
                    "[%s] No se pudo comunicar con coordinador para block %s",
                    WORKER_ID,
                    block_id,
                )
            else:
                logger.info(
                    "[%s] Resultado descartado por el coordinador (status=%s)",
                    WORKER_ID,
                    status,
                )
        else:
            logger.info(
                "[%s] No se encontró solución | block=%s | time=%.2fs | attempts=%d",
                WORKER_ID,
                block_id,
                resultado["processingTime"],
                resultado["intentos"],
            )

            dataResult = {
                "blockId": block_id,
                "workerId": WORKER_ID,
                "processingTime": resultado["processingTime"],
                "hashRate": 0.0,
                "hash": "",
                "result": "",
            }
            enviar_resultado(dataResult)

    except Exception:
        logger.exception("[%s] Error procesando mensaje", WORKER_ID)

    finally:
        try:
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            logger.exception("[%s] Error al ackear el mensaje", WORKER_ID)

        logger.debug("[%s] Esperando bloques...", WORKER_ID)


# -----------------------
# Main
# -----------------------
def main():
    connection = connect_rabbit()
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)
    channel.exchange_declare(
        exchange=exchangeBlock, exchange_type="topic", durable=True
    )

    result = channel.queue_declare("", exclusive=True)
    queue_name = result.method.queue

    channel.queue_bind(exchange=exchangeBlock, queue=queue_name, routing_key="blocks")

    channel.basic_consume(
        queue=queue_name, on_message_callback=on_message_received, auto_ack=False
    )

    logger.info("[%s] Worker CPU listo y esperando bloques...", WORKER_ID)

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
    #print(ejecutar_minero(1, 99_999_999, "0000000", "HOLAMUNDO"))

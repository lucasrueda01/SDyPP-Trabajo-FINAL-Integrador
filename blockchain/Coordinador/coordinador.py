import hashlib
import json
import threading
import sys
import time
import uuid
import logging

from flask import Flask, jsonify, request
import pika
import redis
from google.cloud import storage

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

logger = logging.getLogger("coordinator")

logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("redis").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------
# Redis
# -----------------------
def redisConnect():
    try:
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
        )
        client.ping()
        logger.info("Conectado a Redis")
        return client
    except Exception:
        logger.exception("Error conectando a Redis")
        raise


# -----------------------
# RabbitMQ
# -----------------------
def queueConnect(retries=10, delay=3):
    for i in range(retries):
        try:
            logger.info("Conectando a RabbitMQ (TX)...")
            if settings.RABBIT_URL:
                params = pika.URLParameters(settings.RABBIT_URL)
            else:
                params = pika.ConnectionParameters(
                    host=settings.RABBIT_HOST,
                    credentials=pika.PlainCredentials(
                        settings.RABBIT_USER, settings.RABBIT_PASSWORD
                    ),
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )

            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(
                "QueueTransactions", durable=True
            )  # Cola de transacciones
            channel.queue_declare(
                queue="pool_tasks", durable=True
            )  # Cola de bloques para minar
            logger.info("Conectado a RabbitMQ")
            return connection, channel

        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ no disponible, reintentando...")
            time.sleep(delay)
        except Exception:
            logger.exception("Error inesperado conectando a RabbitMQ")
            time.sleep(delay)

    raise Exception("No se pudo conectar a RabbitMQ")


# -----------------------
# Pool Manager
# -----------------------
def publicar_a_pool_manager(block):
    block_id = block["blockId"]
    props = pika.BasicProperties(
        delivery_mode=2,  # persistente
        message_id=block_id,  # idempotencia
        content_type="application/json",
    )

    channel.basic_publish(
        exchange="",
        routing_key="pool_tasks",
        body=json.dumps(block),
        properties=props,
    )

    logger.info("Bloque %s publicado en pool_tasks", block_id)


# -----------------------
# Helpers
# -----------------------
def encolar(transaction):
    props = pika.BasicProperties(delivery_mode=2)
    channel.basic_publish(
        exchange="",
        routing_key="QueueTransactions",
        body=json.dumps(transaction),
        properties=props,
    )
    logger.info("Transacción encolada")


def validarTransaction(transaction):
    required = ["origen", "destino", "monto"]
    return all(k in transaction and transaction[k] for k in required)


def calculateHash(data):
    h = hashlib.md5()
    h.update(data.encode("utf-8"))
    return h.hexdigest()


# -----------------------
# Redis helpers
# -----------------------
def getUltimoBlock():
    raw = redisClient.lindex("blockchain", 0)
    return json.loads(raw) if raw else None


def existBlock(block_id):
    return redisClient.sismember("block_ids", block_id)


def postBlock(block):
    pipe = redisClient.pipeline()
    pipe.lpush("blockchain", json.dumps(block))
    pipe.sadd("block_ids", block["blockId"])
    pipe.execute()


def gpus_vivas():
    """
    Cuenta cuántos workers GPU están vivos.
    Un worker está vivo si existe la key worker:* (TTL activo).
    """
    count = 0

    for key in redisClient.scan_iter("worker:*"):
        raw = redisClient.get(key)
        if not raw:
            continue

        try:
            worker = json.loads(raw)
        except Exception:
            continue

        if worker.get("type") == "gpu":
            count += 1

    return count


# -----------------------
# Bucket
# -----------------------
def bucketConnect(bucketName):
    client = storage.Client()
    logger.info("Conectado a Google Cloud Storage")
    return client.bucket(bucketName)


def subirBlock(bucket, block):
    blob = bucket.blob(f"block_{block['blockId']}.json")
    blob.upload_from_string(json.dumps(block), content_type="application/json")
    logger.info("Bloque %s subido al bucket", block["blockId"])


def descargarBlock(bucket, blockId):
    blob = bucket.blob(f"block_{blockId}.json")
    return json.loads(blob.download_as_text())


def borrarBlock(bucket, blockId):
    bucket.blob(f"block_{blockId}.json").delete()
    logger.info("Bloque %s eliminado del bucket", blockId)


# -----------------------
# HTTP endpoints
# -----------------------
@app.route("/transaction", methods=["POST"])
def addTransaction():
    tx = request.json
    if not validarTransaction(tx):
        return "Transacción inválida", 400

    encolar(tx)
    return "Transacción aceptada", 202


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "OK"})


@app.route("/solved_task", methods=["POST"])
def receive_solved_task():
    data = request.get_json()
    if not data or not data.get("result"):
        return jsonify({"message": "Resultado invalido"}), 202

    block_id = data["blockId"]
    worker_id = data.get("workerId", "unknown")

    # -----------------------
    # 1) Chequear estado del bloque (IDEMPOTENCIA)
    # -----------------------
    status_key = f"block:{block_id}:status"
    status = redisClient.get(status_key)

    if status == "SEALED":
        # El bloque ya fue resuelto o cerrado
        logger.info(
            "Resultado tardío descartado para bloque %s (status=%s)",
            block_id,
            status,
        )
        return jsonify({"message": "Bloque ya cerrado"}), 202

    # -----------------------
    # 2) Lock corto SOLO para exclusión mutua
    # -----------------------
    claim_key = f"block:{block_id}:claim"

    if not redisClient.set(claim_key, worker_id, nx=True, ex=15):
        return jsonify({"message": "Bloque ya reclamado"}), 202

    try:
        # -----------------------
        # 3) Descargar bloque (ahora es seguro)
        # -----------------------
        block = descargarBlock(bucket, block_id)

        hash_base = block["baseStringChain"] + block["blockchainContent"]
        hash_calc = calculateHash(data["result"] + hash_base)

        if hash_calc != data["hash"]:
            redisClient.delete(claim_key)
            return jsonify({"message": "Hash inválido"}), 202

        # -----------------------
        # 4) Verificar si ya existe en blockchain
        # -----------------------
        if existBlock(block_id):
            redisClient.set(status_key, "SEALED")
            redisClient.delete(claim_key)
            return jsonify({"message": "Bloque ya existe"}), 202

        # -----------------------
        # 5) SELLAR el bloque 
        # -----------------------
        redisClient.set(status_key, "SEALED")

        # -----------------------
        # 6) Construir y persistir el bloque
        # -----------------------
        prev = getUltimoBlock()
        newBlock = {
            "blockId": block_id,
            "hash": data["hash"],
            "hashPrevio": prev["hash"] if prev else None,
            "nonce": data["result"],
            "prefijo": block["prefijo"],
            "transactions": block["transactions"],
            "timestamp": time.time(),
            "baseStringChain": block["baseStringChain"],
            "blockchainContent": calculateHash(
                block["baseStringChain"] + data["hash"]
            ),
        }

        postBlock(newBlock)

        # -----------------------
        # 7) Borrar bloque del bucket)
        # -----------------------
        borrarBlock(bucket, block_id)

        logger.info("Bloque %s agregado a la blockchain", block_id)
        return jsonify({"message": "Bloque aceptado"}), 201

    except Exception:
        # En caso de error inesperado, liberar lock
        redisClient.delete(claim_key)
        logger.exception("Error procesando resultado para bloque %s", block_id)
        return jsonify({"message": "Error interno"}), 500


    except Exception:
        logger.exception("Error procesando bloque resuelto")
        redisClient.delete(claim_key)
        return jsonify({"message": "Error interno"}), 500


# -----------------------
# Background loop
# -----------------------
def processPackages():
    while True:
        txs = []
        for _ in range(settings.MAX_TRANSACTIONS_PER_BLOCK):
            method_frame, _, body = channel.basic_get(queue="QueueTransactions")
            if not method_frame:
                break
            txs.append(json.loads(body))
            channel.basic_ack(method_frame.delivery_tag)

        if txs:
            blockId = str(uuid.uuid4())
            last = getUltimoBlock()

            gpus = gpus_vivas()

            if gpus == 0:
                prefijo = "0" * settings.DIFFICULTY_LOW
            else:
                prefijo = "0" * settings.DIFFICULTY_HIGH

            block = {
                "blockId": blockId,
                "transactions": txs,
                "prefijo": prefijo,
                "baseStringChain": settings.BASE_STRING_CHAIN,
                "blockchainContent": last["blockchainContent"] if last else "0",
                "numMaxRandom": settings.MAX_RANDOM,
            }

            subirBlock(bucket, block)
            publicar_a_pool_manager(block)

        time.sleep(settings.PROCESSING_TIME)


# -----------------------
# Init
# -----------------------
logger.info("Coordinador iniciando")

connection, channel = queueConnect()
redisClient = redisConnect()
bucket = bucketConnect(settings.BUCKET_NAME)

threading.Thread(target=processPackages, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.COORDINADOR_PORT)

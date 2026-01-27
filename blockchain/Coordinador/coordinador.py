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

# Logger propio
logger = logging.getLogger("coordinator")

# Silenciar librerías externas
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("redis").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------
# Conexión a Redis
# -----------------------
def redisConnect():
    try:
        client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)
        # quick ping to verify connection
        client.ping()
        logger.info("Conectado a Redis")
        return client
    except Exception as e:
        logger.exception("Error conectando a Redis")
        raise


# -----------------------
# Conexión a RabbitMQ
# -----------------------
def queueConnect(retries=10, delay=3):
    for i in range(retries):
        try:
            logger.info(f"Conectando a RabbitMQ (intento {i+1})...")
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
            # declare durable queue / exchange
            channel.queue_declare(queue=settings.QUEUE_NAME_TX, durable=True)
            channel.exchange_declare(
                exchange=settings.EXCHANGE_BLOCK, exchange_type="topic", durable=True
            )
            logger.info("Conectado a Rabbit-MQ")
            return connection, channel
        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ no disponible, reintentando...")
            time.sleep(delay)
        except Exception:
            logger.exception("Error inesperado conectando a RabbitMQ")
            time.sleep(delay)

    raise Exception("No se pudo conectar a RabbitMQ")


def encolar(transaction):
    jsonTransaction = json.dumps(transaction)
    props = pika.BasicProperties(delivery_mode=2)  # Para que el mensaje sea persistente
    channel.basic_publish(
        exchange="",
        routing_key=settings.QUEUE_NAME_TX,
        body=jsonTransaction,
        properties=props,
    )
    logger.info(f"Encolada en {settings.QUEUE_NAME_TX}: {transaction}")


def validarTransaction(transaction):
    required = ["origen", "destino", "monto"]
    return all(k in transaction and transaction[k] for k in required)


def calculateHash(data):
    hash_md5 = hashlib.md5()
    hash_md5.update(data.encode("utf-8"))
    return hash_md5.hexdigest()


# -----------------------
# Funciones de Redis
# -----------------------
def getUltimoBlock():
    ultimoBlock = redisClient.lindex("blockchain", 0)
    if ultimoBlock:
        return json.loads(ultimoBlock)
    return None


def existBlock(id):
    try:
        # fast path: set of block ids
        return redisClient.sismember("block_ids", id)
    except Exception:
        # fallback: scan list (compatibility)
        try:
            allBlocks = redisClient.lrange("blockchain", 0, -1)
            for block in allBlocks:
                msg = json.loads(block)
                if "blockId" in msg and msg["blockId"] == id:
                    return True
        except Exception:
            logger.exception("Error verificando existencia de bloque en Redis")
        return False


def postBlock(block):
    try:
        block_json = json.dumps(block)

        pipe = redisClient.pipeline(transaction=True)
        pipe.lpush("blockchain", block_json)

        block_id = block.get("blockId")
        if block_id:
            pipe.sadd("block_ids", block_id)

        pipe.execute()

    except Exception:
        logger.exception("Error posteando bloque en Redis")
        raise


# -----------------------
# Funciones de Bucket
# -----------------------
def bucketConnect(bucketName):
    client = storage.Client()  # usa ADC
    logger.info("Conectado al Bucket de Google Cloud Storage")
    return client.bucket(bucketName)


def subirBlock(bucket, block):
    try:
        blockId = block["blockId"]
        jsonBlock = json.dumps(block)
        fileName = f"block_{blockId}.json"
        blob = bucket.blob(fileName)
        blob.upload_from_string(jsonBlock, content_type="application/json")
        logger.info(
            f"El bloque {blockId} fue subido al bucket con el nombre {fileName}"
        )
    except Exception:
        logger.exception("Error subiendo bloque al bucket")
        raise


def descargarBlock(bucket, blockId):
    try:
        fileName = f"block_{blockId}.json"
        blob = bucket.blob(fileName)
        jsonBlock = blob.download_as_text()
        block = json.loads(jsonBlock)
        logger.info("%s: Descargado del bucket", blockId)
        return block
    except Exception:
        logger.exception("Error descargando bloque %s del bucket", blockId)
        raise


def borrarBlock(bucket, blockId):
    try:
        blob = bucket.blob(f"block_{blockId}.json")
        blob.delete()
        logger.info("Bloque %s borrado correctamente del bucket", blockId)
    except Exception:
        logger.exception("Ocurrió un error al borrar el bloque %s", blockId)


# -----------------------
# HTTP endpoints
# -----------------------
@app.route("/transaction", methods=["POST"])
def addTransaction():
    transaction = request.json
    try:
        if validarTransaction(transaction):
            encolar(transaction)
        else:
            logger.warning("Transaccion no valida recibida: %s", transaction)
            return "Transaccion no recibida", 400
    except Exception:
        logger.exception("Error procesando transaccion")
        return "Transaccion no recibida", 400
    return "Transaccion Recibida", 200


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "OK"})


@app.route("/solved_task", methods=["POST"])
def receive_solved_task():
    # estructura de bloque final
    newBlock = {
        "blockId": None,
        "hash": None,
        "hashPrevio": None,
        "nonce": None,
        "prefijo": None,
        "transactions": None,
        "timestamp": None,
        "blockchainContent": None,
        "baseStringChain": None,
    }

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Ha ocurrido un error"}), 400

    logger.info(
        "Tarea resuelta recibida: %s", {k: data.get(k) for k in ("blockId", "workerId")}
    )

    # validar campo result correctamente
    if not data.get("result"):
        logger.info("Resultado vacío o nulo - descartado")
        return (
            jsonify({"message": "No se encontró un resultado valido -> DESCARTADO"}),
            202,
        )

    # Evitar race condition con claims (solo un worker gana)
    claim_key = f"block:{data['blockId']}:claim"
    worker_id = data.get("workerId", "unknown_worker")
    try:
        # intentamos reclamar: NX -> only set if not exists; EX -> ttl en segundos
        claimed = redisClient.set(claim_key, worker_id, nx=True, ex=60)
    except Exception:
        logger.exception("Error al intentar setear claim en Redis")
        return jsonify({"message": "Error interno"}), 500

    if not claimed:
        logger.info(
            "Bloque %s ya reclamado por otro worker -> descartado", data["blockId"]
        )
        return jsonify({"message": "Bloque ya reclamado -> DESCARTADO"}), 202

    # Si llegamos aca somos el reclamante, si algo falla mas adelante liberar claim
    try:
        block = descargarBlock(bucket, data["blockId"])
    except Exception:
        # liberar claim
        try:
            redisClient.delete(claim_key)
        except Exception:
            logger.exception("No se pudo liberar claim tras fallo descargarBlock")
        return jsonify({"message": "Error interno al descargar bloque"}), 500

    try:
        dataHash = (
            data["result"] + block["baseStringChain"] + block["blockchainContent"]
        )
        hashResult = calculateHash(dataHash)
        timestamp = time.time()
        logger.debug("Hash recibido: %s", data.get("hash"))
        logger.debug("Hash calculado: %s", hashResult)

        if hashResult == data["hash"]:
            logger.info("Hashes coinciden: data valida")

            # Validar si existe el bloque
            if existBlock(block["blockId"]):
                logger.info("Bloque %s ya existe -> descartando", block["blockId"])
                # liberar claim para ser limpios
                redisClient.delete(claim_key)
                return (
                    jsonify({"message": "El bloque ya fue resuelto » DESCARTADO"}),
                    202,
                )

            # calcular blockchainContent y armar bloque definitivo
            blockchainData = block["baseStringChain"] + data["hash"]
            blockchainContent = calculateHash(blockchainData)
            newBlock["blockchainContent"] = blockchainContent

            try:
                ultimoBloque = getUltimoBlock()
            except Exception:
                ultimoBloque = None

            if ultimoBloque:
                logger.info("Hay bloque anterior -> Conectar bloques")
                newBlock["hashPrevio"] = ultimoBloque.get("hash")
            else:
                logger.info("No hay bloque anterior -> Bloque genesis")
                newBlock["hashPrevio"] = None

            newBlock["blockId"] = data["blockId"]
            newBlock["hash"] = data["hash"]
            newBlock["transactions"] = block.get("transactions")
            newBlock["prefijo"] = block.get("prefijo")
            newBlock["baseStringChain"] = block.get("baseStringChain")
            newBlock["timestamp"] = timestamp
            newBlock["nonce"] = data["result"]

            # persistir
            postBlock(newBlock)
            logger.info(
                "Bloque %s validado y agregado a la blockchain", newBlock["blockId"]
            )

            # borrar candidato del bucket
            borrarBlock(bucket, data["blockId"])

            # (opcional) marcar resolved key
            try:
                redisClient.set(
                    f"block:{newBlock['blockId']}:resolved",
                    json.dumps(newBlock),
                    ex=86400,
                )
            except Exception:
                logger.exception("No se pudo marcar bloque como resolved en Redis")

            return (
                jsonify({"message": "Bloque validado -> Agregado a la blockchain"}),
                201,
            )

        else:
            logger.info("Los hashes difieren -> dato inválido")
            # liberar claim
            try:
                redisClient.delete(claim_key)
            except Exception:
                logger.exception("No se pudo liberar claim tras hash mismatch")
            return (
                jsonify({"message": "El Hash recibido es invalido -> DESCARTADO"}),
                202,
            )

    except Exception:
        logger.exception("Error procesando resultado solucion")
        # asegurar liberar claim
        try:
            redisClient.delete(claim_key)
        except Exception:
            logger.exception("No se pudo liberar claim tras exception")
        return jsonify({"message": "Error interno"}), 500


# -----------------------
# Background loop: procesar transacciones y publicar bloques
# -----------------------
def processPackages():
    last_no_tx_log = 0
    while True:
        contadorTransaction = 0
        logger.debug("Buscando transacciones")
        listaTransactions = []
        for _ in range(settings.MAX_TRANSACTIONS_PER_BLOCK):
            try:
                method_frame, _, body = channel.basic_get(queue=settings.QUEUE_NAME_TX)
            except Exception:
                logger.exception("Error leyendo cola RabbitMQ")
                break

            if method_frame:
                contadorTransaction += 1
                try:
                    listaTransactions.append(json.loads(body))
                except Exception:
                    logger.exception("Error decodificando transaccion: %s", body)
                logger.debug("Desencolada transaccion")
                logger.debug("Transaccion: %s", body)
                channel.basic_ack(method_frame.delivery_tag)
            else:
                # Info periodica si no hay transacciones
                now = time.time()
                if now - last_no_tx_log > 20:  # cada 20s una vez
                    logger.info(
                        "No hay transacciones - count desencoladas: %d",
                        contadorTransaction,
                    )
                    last_no_tx_log = now
                break

        if listaTransactions:
            blockId = str(uuid.uuid4())
            block = {
                "blockId": blockId,
                "transactions": listaTransactions,
                "prefijo": settings.PREFIX,
                "baseStringChain": settings.BASE_STRING_CHAIN,
                "blockchainContent": (
                    getUltimoBlock()["blockchainContent"] if getUltimoBlock() else "0"
                ),
                "numMaxRandom": settings.MAX_RANDOM,
            }

            logger.info(
                "Preparando bloque candidato %s (txs=%d)",
                blockId,
                len(listaTransactions),
            )

            # subir y publicar
            try:
                subirBlock(bucket, block)
            except Exception:
                logger.exception("Error subiendo bloque candidato al bucket")
                # no publicar si no sube
                time.sleep(settings.TIMER)
                continue

            try:
                # publicar como mensaje durable
                props = pika.BasicProperties(delivery_mode=2)
                channel.basic_publish(
                    exchange=settings.EXCHANGE_BLOCK,
                    routing_key="blocks",
                    body=json.dumps(block),
                    properties=props,
                )
                logger.info("Bloque %s publicado al exchange", blockId)
            except Exception:
                logger.exception("Error publicando bloque al exchange")
        time.sleep(settings.TIMER)


# -----------------------
# Inicializacion
# -----------------------
logger.info("Escuchando en %s:%s", settings.COORDINADOR_HOST, settings.COORDINADOR_PORT)
connection, channel = queueConnect()
redisClient = redisConnect()
bucket = bucketConnect(settings.BUCKET_NAME)
status_thread = threading.Thread(target=processPackages, daemon=True)
status_thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.COORDINADOR_PORT)

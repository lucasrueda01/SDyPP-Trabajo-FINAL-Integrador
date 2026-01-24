import hashlib
import json
import threading
from flask import Flask, jsonify, request
import pika
import redis
import time
from google.cloud import storage
import uuid
import config.settings as settings

app = Flask(__name__)


# Conexion a Redis
def redisConnect():
    client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)
    print("[x] Conectado a Redis")
    return client


def queueConnect(retries=10, delay=3):
    for i in range(retries):
        try:
            print(f"[x] Conectando a RabbitMQ (intento {i+1})...")
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=settings.RABBIT_HOST,
                    credentials=pika.PlainCredentials(
                        settings.RABBIT_USER,
                        settings.RABBIT_PASSWORD
                    ),
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
            )
            channel = connection.channel()
            channel.queue_declare(queue=settings.QUEUE_NAME_TX)
            channel.exchange_declare(
                exchange=settings.EXCHANGE_BLOCK,
                exchange_type="topic",
                durable=True
            )
            print("[x] Conectado a Rabbit-MQ")
            return connection, channel

        except pika.exceptions.AMQPConnectionError:
            print("[!] RabbitMQ no disponible, reintentando...")
            time.sleep(delay)

    raise Exception("No se pudo conectar a RabbitMQ")


"""
def bucketConnect(bucketName, credentialPath):
    bucketclient = storage.Client.from_service_account_json(credentialPath)
    bucket = bucketclient.bucket(bucketName)
    return bucket
"""

def bucketConnect(bucketName):
    client = storage.Client()   # usa ADC
    return client.bucket(bucketName)

def encolar(transaction):

    jsonTransaction = json.dumps(transaction)
    channel.basic_publish(
        exchange="", routing_key=settings.QUEUE_NAME_TX, body=jsonTransaction
    )
    print(f"[x] Se enconlo en {settings.QUEUE_NAME_TX}: {transaction}")


# Validar que lo que me haya llegado es una transaccion
# {'origen': 'idOrigen'
#  'destino': 'idDestino'
#  'monto': 'monto'}


def validarTransaction(transaction):
    required = ["origen", "destino", "monto"]
    return all(k in transaction and transaction[k] for k in required)


def calculateHash(data):
    hash_md5 = hashlib.md5()
    hash_md5.update(data.encode("utf-8"))
    return hash_md5.hexdigest()


# --- Metodos Redis --- #


def getUltimoBlock():

    ultimoBlock = client.lindex("blockchain", 0)
    if ultimoBlock:
        return json.loads(ultimoBlock)
    return None


def existBlock(id):
    allBlocks = client.lrange("blockchain", 0, -1)

    for block in allBlocks:
        msg = json.loads(block)
        if "blockId" in msg and msg["blockId"] == id:
            return True
    return False


def postBlock(block):
    blockJson = json.dumps(block)
    client.lpush("blockchain", blockJson)


# --- TERMINAN METODOS REDIS --- #


def subirBlock(bucket, block):  # bucket, block

    blockId = block["blockId"]
    jsonBlock = json.dumps(block)

    fileName = f"block_{blockId}.json"

    # Crear un blob (objeto en el bucket) con el nombre deseado
    blob = bucket.blob(fileName)

    # Subir la imagen al blob
    blob.upload_from_string(jsonBlock, content_type="application/json")

    print(f"[x] El bloque {blockId} fue subido al bucket con el nombre de {fileName}")


def descargarBlock(bucket, blockId):

    # Nombre del archivo en bucket
    fileName = f"block_{blockId}.json"
    # Obtener el blob (archivo) del bucket
    blob = bucket.blob(fileName)
    # Descargamos del bucket
    jsonBlock = blob.download_as_text()

    # Serializamos
    block = json.loads(jsonBlock)

    print(f"[x] {blockId} Descargo del Bucket")
    return block


def borrarBlock(bucket, blockId):
    blob = bucket.blob(f"block_{blockId}.json")
    try:
        blob.delete()
        print("Bloque borrado correctamente")
    except Exception as e:
        print(f"Ocurrió un error al borrar el bloque: {e}")


@app.route("/transaction", methods=["POST"])
def addTransaction():

    transaction = request.json

    try:
        if validarTransaction(transaction):
            print("OK")
            encolar(transaction)  # Encolar

    except Exception as e:
        print("[x] La transaccion no es valida")
        print(e)
        return "Transaccion no recibida", 400

    return "Transaccion Recibida", 200


@app.route("/status", methods=["GET"])
def status():
    mensaje = jsonify({"status": "OK"})
    return mensaje


@app.route("/solved_task", methods=["POST"])
def receive_solved_task():

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

    # Procesa los datos recibidos
    if data:
        print(f"Received data: {data}")
        if data["result"] and data["result"] == "":
            return (
                jsonify({"message": "No se encontró un resultado valido -> DESCARTADO"}),
                202,
            )
        bucket = bucketConnect(settings.BUCKET_NAME)
        block = descargarBlock(bucket, data["blockId"])
        dataHash = (
            data["result"] + block["baseStringChain"] + block["blockchainContent"]
        )
        hashResult = calculateHash(dataHash)
        timestamp = time.time()
        print(f"[x] Hash recibido: {data['hash']}")
        print(f"[x] Hash calculado: {hashResult}")
        print("")

        if hashResult == data["hash"]:
            print("[x] Los Hash son iguales » Data valida.")

            # Validar si existe el bloque
            if existBlock(block["blockId"]):
                print("[x] Existe Bloque » Descartar")
                return (
                    jsonify({"message": "El bloque ya fue resuelto » DESCARTADO"}),
                    202,
                )
            else:
                print("[x] No existe bloque » Proceder")
                print("")

                # Calcular blockchainContent
                blockchainData = (
                    block["baseStringChain"] + data["hash"]
                )  # H('A2F8' + Hash del worker)
                blockchainContent = calculateHash(blockchainData)
                newBlock["blockchainContent"] = blockchainContent
                print(f"[x] Blockchain Content: {blockchainContent}")

                # Obtener ultimo bloque

                try:
                    ultimoBloque = getUltimoBlock()
                except:
                    ultimoBloque = None

                if ultimoBloque != None:
                    print("[x] Hay bloque anterior -> Conectar bloques")

                    # Conectar los bloques
                    newBlock["hashPrevio"] = ultimoBloque["hash"]
                    print(f"[x] Hash del ultimo bloque: {ultimoBloque['hash']}")

                else:
                    print("[x] No hay bloque anterior -> Bloque genesis")
                    newBlock["hashPrevio"] = None

            # Armamos bloque

            newBlock["blockId"] = data["blockId"]
            newBlock["hash"] = data["hash"]
            newBlock["transactions"] = block["transactions"]
            newBlock["prefijo"] = block["prefijo"]
            newBlock["baseStringChain"] = block["baseStringChain"]
            newBlock["timestamp"] = timestamp
            newBlock["nonce"] = data["result"]

            postBlock(newBlock)
            print("[x] Bloque validado » Agregado a la blockchain")
            borrarBlock(bucket, data["blockId"])

            return (
                jsonify({"message": "Bloque validado -> Agregado a la blockchain"}),
                201,
            )

        else:
            print("[x] Los Hash son distintos -> Dato invalido")
            return (
                jsonify({"message": "El Hash recibido es invalido -> DESCARTADO"}),
                202,
            )

    else:
        # Si no se reciben datos, responde con un error
        return jsonify({"status": "error", "message": "Ha ocurrido un error"}), 400


def processPackages():
    while True:
        contadorTransaction = 0
        print("[x] Buscando Transacciones")
        print("---------------------------")
        listaTransactions = []
        for _ in range(settings.MAX_TRANSACTIONS_PER_BLOCK):
            method_frame, _, body = channel.basic_get(queue=settings.QUEUE_NAME_TX)
            if method_frame:
                contadorTransaction = contadorTransaction + 1
                listaTransactions.append(json.loads(body))
                print(f"[x] Desencole una transaccion")
                print(f"[x] Transaccion: {body}")
                print("")
                channel.basic_ack(method_frame.delivery_tag)
            else:
                print("[x] No hay transacciones")
                print(
                    f"[x] Cantidad de trasacciones desencoladas: {contadorTransaction}"
                )
                print("")
                break

        if listaTransactions:  # Armar bloque
            print("")

            # blockId = str(random.randint(0, maxRandom))
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

            print(f"blockchainContent: {block['blockchainContent']}")

            # Me conecto al bucket y guardo el bloque
            bucket = bucketConnect(settings.BUCKET_NAME)
            subirBlock(bucket, block)

            # Publicar el bloque en el Topic
            channel.basic_publish(
                exchange=settings.EXCHANGE_BLOCK,
                routing_key="blocks",
                body=json.dumps(block),
            )
            print(f"[x] Bloque {blockId} enviado")
            print("")

        time.sleep(settings.TIMER)


# Conectamos a la cola
print(f"Escuchando en {settings.COORDINADOR_HOST}:{settings.COORDINADOR_PORT}")
connection, channel = queueConnect()
client = redisConnect()
status_thread = threading.Thread(target=processPackages)
status_thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.COORDINADOR_PORT)
    

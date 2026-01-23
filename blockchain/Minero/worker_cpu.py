import pika
import json
import hashlib
import random
import requests
import time
import config.settings as settings

hostRabbit = settings.RABBIT_HOST
exchangeBlock = settings.EXCHANGE_BLOCK
hostCoordinador = settings.COORDINADOR_HOST
puertoCoordinador = settings.COORDINADOR_PORT
rabbitUser = settings.RABBIT_USER
rabbitPassword = settings.RABBIT_PASSWORD

# Permite métricas y saber quién resolvió el bloque
WORKER_ID = f"cpu-{random.randint(1000, 9999)}"

#   block = {
#                "blockId": blockId,
#               "transactions": listaTransactions,
#              "prefijo": '000',
#             "baseStringChain": "A3F8",
#            "blockchainContent": "contenido",
#           "numMaxRandom": maxRandom
#      }


def calculateHash(data):
    hash_md5 = hashlib.md5()
    hash_md5.update(data.encode("utf-8"))
    return hash_md5.hexdigest()


def enviar_resultado(data: dict) -> int | None:
    """
    - Devuelve el status code del coordinador
    - Permite saber si el bloque fue aceptado o descartado
    """
    url = f"http://{hostCoordinador}:{puertoCoordinador}/solved_task"
    try:
        response = requests.post(url, json=data, timeout=5)
        print("Post response:", response.text)
        return response.status_code
    except requests.exceptions.RequestException as e:
        print("Failed to send POST request:", e)
        return None


def on_message_received(channel, method, properties, body):
    data = json.loads(body)
    print(f"[{WORKER_ID}] Bloque recibido: {data}")
    print("")

    encontrado = False
    intentos = 0
    nonce = 0
    startTime = time.time()

    print("## Iniciando Minero CPU ##")

    while not encontrado and nonce <= data["numMaxRandom"]:
        intentos += 1
        nonce += 1
        nonce_str = str(nonce)

        hashCalculado = calculateHash(
            nonce_str + data["baseStringChain"] + data["blockchainContent"]
        )

        if hashCalculado.startswith(data["prefijo"]):
            encontrado = True
            processingTime = time.time() - startTime

            hash_rate = intentos / processingTime if processingTime > 0 else 0

            dataResult = {
                "blockId": data["blockId"],
                "workerId": WORKER_ID,
                "processingTime": processingTime,
                "hashRate": hash_rate,
                "hash": hashCalculado,
                "result": nonce_str,
            }

            print(f"[x] Hash con el prefijo {data['prefijo']} encontrado")
            print(f"[x] HASH: {hashCalculado}")
            print("")
            status = enviar_resultado(dataResult)
            if status == 201:
                print("[x] Bloque aceptado por el coordinador")
            else:
                print("[x] Resultado descartado por el coordinador")

            print(
                f"Resultado: {nonce_str}",
                f"[x] Tiempo: {processingTime:.2f}s | "
                f"Intentos: {intentos} | "
                f"HashRate: {hash_rate:.2f} H/s",
            )
            print("")
            break

    if not encontrado:
        print("[x] No se encontró solución en el rango asignado")
        processingTime = time.time() - startTime

        dataResult = {
            "blockId": data["blockId"],
            "workerId": WORKER_ID,
            "processingTime": processingTime,
            "hashRate": 0,
            "hash": "",
            "result": "",
        }

        enviar_resultado(dataResult)

    # ACK solo cuando el worker terminó
    channel.basic_ack(delivery_tag=method.delivery_tag)
    print(f"[{WORKER_ID}] Esperando bloques...")


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=hostRabbit,
            credentials=pika.PlainCredentials(rabbitUser, rabbitPassword),
        )
    )
    channel = connection.channel()
    channel.exchange_declare(
        exchange=exchangeBlock, exchange_type="topic", durable=True
    )
    result = channel.queue_declare("", exclusive=True)
    queue_name = result.method.queue

    channel.queue_bind(exchange=exchangeBlock, queue=queue_name, routing_key="blocks")
    channel.basic_consume(
        queue=queue_name, on_message_callback=on_message_received, auto_ack=False
    )
    print(f"[{WORKER_ID}] Esperando bloques...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Worker detenido por usuario")
        connection.close()


if __name__ == "__main__":
    main()

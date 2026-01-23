import time
import pika
import requests
import json
from minero import minero_gpu
import config.settings as settings
import random

# Configuracion
hostRabbit = settings.RABBIT_HOST
exchangeBlock = settings.EXCHANGE_BLOCK
hostCoordinador = settings.COORDINADOR_HOST
puertoCoordinador = settings.COORDINADOR_PORT
rabbitUser = settings.RABBIT_USER
rabbitPassword = settings.RABBIT_PASSWORD

# ID único del worker GPU
WORKER_ID = f"gpu-{random.randint(1000,9999)}"


# Enviar el resultado al coordinador para verificar que el resultado es correcto
def enviar_resultado(data):
    url = f"http://{hostCoordinador}:{puertoCoordinador}/solved_task"
    try:
        response = requests.post(url, json=data, timeout=5)
        print("Respuesta del coordinador:", response.text)
        return response.status_code
    except Exception as e:
        print("Fallo al enviar el post:", e)
        return None


# Minero
def on_message_received(ch, method, properties, body):
    try:
        data = json.loads(body)
        print(f"[{WORKER_ID}] Bloque recibido: {data['blockId']}")
        print("")
        from_val = 1
        to_val = data["numMaxRandom"]
        prefix = data["prefijo"]
        hash_base = data["baseStringChain"] + data["blockchainContent"]

        startTime = time.time()

        print("## Iniciando Minero GPU ##")
        resultado = minero_gpu.ejecutar_minero(from_val, to_val, prefix, hash_base)
        resultado = json.loads(resultado)

        processingTime = time.time() - startTime

        if not resultado.get("hash_md5_result"):
            print("[x] GPU no encontró solución en el rango")
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

        intentos = resultado.get("intentos", data["numMaxRandom"])
        hash_rate = intentos / processingTime if processingTime > 0 else 0

        dataResult = {
            "blockId": data["blockId"],
            "workerId": WORKER_ID,
            "processingTime": processingTime,
            "hashRate": hash_rate,
            "hash": resultado["hash_md5_result"],
            "result": str(resultado["numero"]),
        }

        status = enviar_resultado(dataResult)

        if status == 201:
            print("[x] Bloque aceptado por el coordinador")
        else:
            print("[x] Resultado descartado por el coordinador")

        print(
            f"[x] Resultado: {resultado['numero']} | "
            f"Tiempo: {processingTime:.2f}s | "
            f"Intentos: {intentos} | "
            f"HashRate: {hash_rate:.2f} H/s"
        )
        print("")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("[x] Error en worker GPU:", e)
        ch.basic_ack(delivery_tag=method.delivery_tag)


# Conexion con rabbit al topico y comienza a ser consumidor
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
    print(f"[{WORKER_ID}] Worker GPU esperando bloques...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Worker detenido por usuario")
        connection.close()


if __name__ == "__main__":
    main()

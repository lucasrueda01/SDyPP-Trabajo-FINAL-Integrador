import time
import pika
import requests
import json
import minero_gpu
import config.settings as settings

# Configuracion
hostRabbit = settings.RABBIT_HOST
exchangeBlock = settings.EXCHANGE_BLOCK
hostCoordinador = settings.COORDINADOR_HOST
puertoCoordinador = settings.COORDINADOR_PORT
rabbitUser = settings.RABBIT_USER
rabbitPassword = settings.RABBIT_PASSWORD

# Enviar el resultado al coordinador para verificar que el resultado es correcto
def enviar_resultado(data):
    url = f"http://{hostCoordinador}:{puertoCoordinador}/solved_task"
    try:
        requests.post(url, json=data)
        print("Resolucion enviada al Coordinador!")
    except Exception as e:
        print("Fallo al enviar el post:", e)


# Minero
def minero(ch, method, properties, body):
    try:
        data = json.loads(body)
        print(f"Bloque {data} recibido")
        startTime = time.time()
        print("Minero comenzado!")
        # Salida: {"numero": 278310, "hash_md5_result": "00000879bbaa8f7bdd50fac39acefd64"}
        hash_val = data["baseStringChain"] + data["blockchainContent"]
        resultado = minero_gpu.ejecutar_minero(
            1, data["numMaxRandom"], data["prefijo"], hash_val
        )

        resultado = json.loads(resultado)
        processingTime = time.time() - startTime
        dataResult = {
            "blockId": data["blockId"],
            "processingTime": processingTime,
            "hash": resultado["hash_md5_result"],
            "result": str(resultado["numero"]),
        }

        enviar_resultado(dataResult)
        # Confirmo con un ACK que lo resolvi
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Resultado encontrado y enviado con el ID Bloque {data['blockId']}")

    except Exception as e:
        print("JSON inválido:", e)
        print("Contenido:", resultado)
        return


# Conexion con rabbit al topico y comienza a ser consumidor
def main():

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=hostRabbit, credentials=pika.PlainCredentials(rabbitUser, rabbitPassword)
        )
    )
    channel = connection.channel()
    channel.exchange_declare(
        exchange=exchangeBlock, exchange_type="topic", durable=True
    )
    result = channel.queue_declare("", exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(exchange=exchangeBlock, queue=queue_name, routing_key="blocks")
    channel.basic_consume(queue=queue_name, on_message_callback=minero, auto_ack=False)
    print("Esperando mensajes. Para salir pulse CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        connection.close()
        print("Conexion cerrada")


if __name__ == "__main__":
    main()

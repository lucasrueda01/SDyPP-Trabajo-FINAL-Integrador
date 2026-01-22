import pika
import json
import hashlib
import random
import requests
import time

hostRabbit = 'localhost'
queueNameTx = 'QueueTransactions'
exchangeBlock = 'ExchangeBlock'

def calculateHash(data):
    hash_md5 = hashlib.md5()
    hash_md5.update(data.encode('utf-8'))
    return hash_md5.hexdigest()

def sendResult(data):
    url = "http://localhost:5000/solved_task"
    try:
        response = requests.post(url, json=data)
        print("Post response:", response.text)
    except requests.exceptions.RequestException as e:
        print("Failed to send POST request:", e)

#   block = {
#                "blockId": blockId,
 #               "transactions": listaTransactions,
  #              "prefijo": '000',
   #             "baseStringChain": "A3F8",
    #            "blockchainContent": "contenido",
     #           "numMaxRandom": maxRandom 
      #      }


def on_message_received(channel, method, properties, body):
    data = json.loads(body)
    print(f"Message {data} received")
    print('')

    encontrado = False
    intentos   = 0
    startTime  = time.time()

    print("## Iniciando Minero ##")

    while not encontrado:
        intentos = intentos + 1
        randomNumber = str(random.randint(1, data['numMaxRandom']))
        
        hashCalculado = calculateHash(randomNumber + data['baseStringChain'] + data['blockchainContent'])
        if hashCalculado.startswith(data['prefijo']):
            encontrado = True
            processingTime = time.time() - startTime

            dataResult = {
                'blockId': data['blockId'],
                'processingTime': processingTime,
                'hash': hashCalculado,
                'result': randomNumber   
            }

            print(f"[x] Hash con el prefijo {data['prefijo']} encontrado")
            print(f"[x] HASH: {hashCalculado}")
            print('')
            sendResult(dataResult) 
            print(f"Result found and posted for block ID {data['blockId']} in {processingTime:.2f} seconds in {intentos} intentos")
            print(f"Resultado: {randomNumber}")
            print('')
            
        if intentos > data['numMaxRandom']:
            print("No se encontro el resultado")
            processingTime = time.time() - startTime
            dataResult = {
                'blockId': data['blockId'],
                'processingTime': processingTime,
                'hash': "",
                'result': ""   
            }
            sendResult(dataResult)
            break
        
    channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=hostRabbit, credentials=pika.PlainCredentials("guest", "guest"))
    )
    channel = connection.channel()
    channel.exchange_declare(exchange= exchangeBlock, exchange_type='topic', durable=True)
    result = channel.queue_declare('', exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(exchange= exchangeBlock, queue=queue_name, routing_key='blocks')
    channel.basic_consume(queue=queue_name, on_message_callback=on_message_received, auto_ack=False)
    print('Waiting for messages. To exit press CTRL+C')
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Consumption stopped by user.")
        connection.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()
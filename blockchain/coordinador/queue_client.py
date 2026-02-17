import json
import time
import pika
import logging
import config.settings as settings

logger = logging.getLogger("coordinator")


def queueConnect(delay=5):
    while True:
        try:
            logger.info("Conectando a RabbitMQ...")

            if settings.RABBIT_URL:
                params = pika.URLParameters(settings.RABBIT_URL)
            else:
                params = pika.ConnectionParameters(
                    host=settings.RABBIT_HOST,
                    credentials=pika.PlainCredentials(
                        settings.RABBIT_USER,
                        settings.RABBIT_PASSWORD,
                    ),
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )

            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            channel.queue_declare(
                "QueueTransactions",
                durable=True,
                arguments={"x-queue-type": "quorum"},
            )

            channel.queue_declare(
                queue="pool_tasks",
                durable=True,
                arguments={"x-queue-type": "quorum"},
            )

            logger.info("Conectado a RabbitMQ correctamente")
            return connection, channel

        except Exception as e:
            logger.warning(
                "RabbitMQ no disponible (%s). Reintentando en %ss...",
                e,
                delay,
            )
            time.sleep(delay)


def encolar(transaction):
    connection, channel = queueConnect()
    try:
        props = pika.BasicProperties(delivery_mode=2)
        logger.debug(f"Encolando transacción: {transaction}")
        channel.basic_publish(
            exchange="",
            routing_key="QueueTransactions",
            body=json.dumps(transaction),
            properties=props,
        )
    finally:
        connection.close()


def publicar_a_pool_manager(block, channel):
    props = pika.BasicProperties(
        delivery_mode=2,
        message_id=block["blockId"],
        content_type="application/json",
    )

    channel.basic_publish(
        exchange="",
        routing_key="pool_tasks",
        body=json.dumps(block),
        properties=props,
    )
    logger.debug(f"Bloque {block['blockId']} publicado a pool manager")

import json
import time
import pika
import logging
import config.settings as settings

logger = logging.getLogger("coordinator")


def queueConnect(retries=10, delay=3):
    for _ in range(retries):
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

            return connection, channel

        except Exception:
            logger.warning("RabbitMQ no disponible, reintentando...")
            time.sleep(delay)

    raise Exception("No se pudo conectar a RabbitMQ")


def encolar(transaction):
    connection, channel = queueConnect()
    try:
        props = pika.BasicProperties(delivery_mode=2)

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

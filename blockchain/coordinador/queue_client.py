import json
import time
import pika
import logging
import config.settings as settings

logger = logging.getLogger("coordinator")

_connection = None
_channel = None


def init_publisher():
    global _connection, _channel

    if (
        _connection is None
        or _connection.is_closed
        or _channel is None
        or _channel.is_closed
    ):
        _connection, _channel = queueConnect()


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
                    heartbeat=30,
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
    global _connection, _channel
    if _connection is None or _connection.is_closed:
        init_publisher()
    try:
        props = pika.BasicProperties(delivery_mode=2)
        _channel.basic_publish(
            exchange="",
            routing_key="QueueTransactions",
            body=json.dumps(transaction),
            properties=props,
        )
        logger.debug(f"Encolando transacción: {transaction}")
    except Exception:
        logger.warning("Error publicando, reconectando...")
        init_publisher()
        _channel.basic_publish(
            exchange="",
            routing_key="QueueTransactions",
            body=json.dumps(transaction),
            properties=props,
        )


def publicar_a_pool_manager(block):
    global _connection, _channel

    props = pika.BasicProperties(
        delivery_mode=2,
        message_id=block["blockId"],
        content_type="application/json",
    )

    try:
        # 🔎 Validar conexión y canal
        init_publisher()

        _channel.basic_publish(
            exchange="",
            routing_key="pool_tasks",
            body=json.dumps(block),
            properties=props,
        )

        logger.debug(f"Bloque {block['blockId']} publicado a pool manager")

    except Exception as e:
        logger.warning(f"Error publicando bloque {block['blockId']}: {e}")
        logger.info("Reintentando publicación tras reconectar...")

        # 🔁 Forzar reconexión
        _connection, _channel = queueConnect()

        # 🔁 Reintento
        _channel.basic_publish(
            exchange="",
            routing_key="pool_tasks",
            body=json.dumps(block),
            properties=props,
        )

        logger.info(f"Bloque {block['blockId']} publicado tras reconexión")

import time
import pika
import logging
import config.settings as settings

logger = logging.getLogger("pool-manager")


def queue_connect(retries=10, delay=3):
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
                    heartbeat=30,
                    blocked_connection_timeout=120,
                )

            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            # Exchanges
            channel.exchange_declare(
                exchange="blocks_cooperative",
                exchange_type="fanout",
                durable=True,
            )

            channel.exchange_declare(
                exchange="blocks_competitive",
                exchange_type="fanout",
                durable=True,
            )

            # Pool queue
            channel.queue_declare(
                queue="pool_tasks", durable=True, arguments={"x-queue-type": "quorum"}
            )

            # Cola compartida de workers
            channel.queue_declare(
                queue="queue.workers",
                durable=True,
                arguments={"x-queue-type": "quorum"},
            )

            channel.queue_bind(
                exchange="blocks_cooperative",
                queue="queue.workers",
            )

            logger.info("Conectado a RabbitMQ en %s:%s", params.host, params.port)
            return connection, channel

        except Exception:
            logger.warning(
                "RabbitMQ no disponible, reintentando (%s/%s)...",
                i + 1,
                retries,
            )
            time.sleep(delay)

    raise Exception("No se pudo conectar a RabbitMQ")


def safe_publish(channel, exchange, routing_key, body):
    channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=body,
        properties=pika.BasicProperties(delivery_mode=2),
    )

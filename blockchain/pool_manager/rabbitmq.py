import time
import pika
import logging
import config.settings as settings

logger = logging.getLogger("pool-manager")


def queue_connect(retries=10, delay=3):
    for i in range(retries):
        try:
            if settings.RABBIT_URL:
                params = pika.URLParameters(settings.RABBIT_URL)
            else:
                params = pika.ConnectionParameters(
                    host=settings.RABBIT_HOST,
                    port=int(settings.RABBIT_PORT),
                    virtual_host=settings.RABBIT_VHOST,
                    credentials=pika.PlainCredentials(
                        settings.RABBIT_USER,
                        settings.RABBIT_PASSWORD,
                    ),
                    heartbeat=600,
                )

            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            # Exchanges
            channel.exchange_declare(
                exchange="blocks_cooperative",
                exchange_type="topic",
                durable=True,
            )

            channel.exchange_declare(
                exchange="blocks_competitive",
                exchange_type="fanout",
                durable=True,
            )

            # Pool queue
            channel.queue_declare(queue="pool_tasks", durable=True)

            # ===== DLQ =====
            dlx_name = "dlx.tasks"
            gpu_ttl = 60000  # ms

            channel.exchange_declare(
                exchange=dlx_name,
                exchange_type="fanout",
                durable=True,
            )

            channel.queue_declare(queue="queue.dlq", durable=True)
            channel.queue_bind(exchange=dlx_name, queue="queue.dlq")

            # ===== WORKER QUEUES (CANÓNICAS) =====

            # CPU (simple)
            channel.queue_declare(
                queue="queue.cpu",
                durable=True,
            )

            # GPU (TTL + DLQ DEFINITIVO)
            channel.queue_declare(
                queue="queue.gpu",
                durable=True,
                arguments={
                    "x-message-ttl": gpu_ttl,
                    "x-dead-letter-exchange": dlx_name,
                    "x-dead-letter-routing-key": "dlq",  # <-- CLAVE
                },
            )
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

import pika
import logging
import config.settings as settings

logger = logging.getLogger("pool-manager")


def queue_connect():
    if settings.RABBIT_URL:
        params = pika.URLParameters(settings.RABBIT_URL)
    else:
        params = pika.ConnectionParameters(
            host=settings.RABBIT_HOST,
            credentials=pika.PlainCredentials(
                settings.RABBIT_USER, settings.RABBIT_PASSWORD
            ),
            heartbeat=600,
        )

    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.exchange_declare(
        exchange="blocks_cooperative", exchange_type="topic", durable=True
    )
    channel.exchange_declare(
        exchange="blocks_competitive", exchange_type="fanout", durable=True
    )

    channel.queue_declare(queue="pool_tasks", durable=True)

    dlx_name = "dlx.tasks"
    channel.exchange_declare(exchange=dlx_name, exchange_type="fanout", durable=True)
    channel.queue_declare(queue="queue.dlq", durable=True)
    channel.queue_bind(exchange=dlx_name, queue="queue.dlq")

    channel.queue_declare(
        queue="queue.gpu",
        durable=True,
        arguments={"x-message-ttl": 60000, "x-dead-letter-exchange": dlx_name},
    )

    channel.queue_declare(queue="queue.cpu", durable=True)

    return connection, channel


def safe_publish(channel, exchange, routing_key, body):
    channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=body,
        properties=pika.BasicProperties(delivery_mode=2),
    )

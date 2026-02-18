import json
import time
import logging
from rabbitmq import queue_connect
from dispatcher import dispatch_to_workers
from redis_workers import get_alive_workers
import traceback


logger = logging.getLogger("pool-manager")


def start_pool_consumer(redis_client):
    while True:
        try:
            logger.info("Iniciando pool consumer...")
            connection, channel_pool = queue_connect()
            channel_pool.basic_qos(prefetch_count=1)

            def on_message(ch, method, properties, body):
                try:
                    block = json.loads(body)
                    
                    alive, _ = get_alive_workers(redis_client)
                    ok = dispatch_to_workers(block, alive, channel_pool)

                    if ok:
                        ch.basic_ack(method.delivery_tag)
                    else:
                        ch.basic_nack(method.delivery_tag, requeue=True)

                except Exception:
                    logger.exception("Error procesando mensaje")
                    ch.basic_nack(method.delivery_tag, requeue=True)

            channel_pool.basic_consume(
                queue="pool_tasks",
                on_message_callback=on_message,
                auto_ack=False,
            )

            logger.info("Pool Consumer iniciado.")
            channel_pool.start_consuming()

        except Exception:
            logger.exception("Consumer perdió conexión, reconectando en 5s...")
            time.sleep(5)


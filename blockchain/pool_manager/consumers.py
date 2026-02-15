import json
import time
import logging
from rabbitmq import queue_connect
from dispatcher import dispatch_to_workers
from redis_workers import get_alive_workers
import traceback


logger = logging.getLogger("pool-manager")

def start_pool_consumer(redis_client):
    try:
        connection, channel_pool = queue_connect()
        channel_pool.basic_qos(prefetch_count=1)

        def on_message(ch, method, properties, body):

            block = json.loads(body)
            alive, _ = get_alive_workers(redis_client)

            ok = dispatch_to_workers(block, alive, channel_pool)

            logger.info(
                "Recibido bloque %s desde pool_tasks",
                block["blockId"],
            )

            if ok:
                ch.basic_ack(method.delivery_tag)
                logger.info("Bloque %s despachado correctamente", block["blockId"])
            else:
                time.sleep(5)
                ch.basic_nack(method.delivery_tag, requeue=True)

        channel_pool.basic_consume(
            queue="pool_tasks",
            on_message_callback=on_message,
            auto_ack=False,
        )
        logger.info("Pool Consumer iniciado, esperando mensajes...")
        channel_pool.start_consuming()
    except Exception as e:
        traceback.print_exc()

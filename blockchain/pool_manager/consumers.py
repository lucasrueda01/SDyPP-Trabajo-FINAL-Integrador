import json
import time
import logging
from rabbitmq import queue_connect
from dispatcher import dispatch_to_workers
from redis_workers import get_alive_workers
from fragmenter import fragmentar
from rabbitmq import safe_publish
import metrics


logger = logging.getLogger("pool-manager")


def start_pool_consumer(redis_client):
    connection, channel = queue_connect()
    channel.basic_qos(prefetch_count=1)

    def on_message(ch, method, properties, body):

        block = json.loads(body)
        alive, _ = get_alive_workers(redis_client)

        ok = dispatch_to_workers(block, alive, channel)

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

    channel.basic_consume(
        queue="pool_tasks",
        on_message_callback=on_message,
        auto_ack=False,
    )

    channel.start_consuming()


def start_dlq_consumer(redis_client):
    connection, channel = queue_connect()
    channel.basic_qos(prefetch_count=1)

    def on_dlq(ch, method, properties, body):
        block = json.loads(body)
        logger.warning(
            "DLQ: mensaje expirado blockId=%s",
            block.get("blockId"),
        )

        alive, _ = get_alive_workers(redis_client)
        cpu_workers = [w for w in alive if w["type"] == "cpu"]

        _, cpu_payloads = fragmentar(block, 0, len(cpu_workers))

        logger.info(
            "DLQ: reasignando bloque %s a %d CPUs",
            block.get("blockId"),
            len(cpu_payloads),
        )

        for payload in cpu_payloads:
            safe_publish(
                channel,
                "blocks_cooperative",
                "blocks.cpu",
                json.dumps(payload),
            )

        ch.basic_ack(method.delivery_tag)

    channel.basic_consume(
        queue="queue.dlq",
        on_message_callback=on_dlq,
        auto_ack=False,
    )

    channel.start_consuming()
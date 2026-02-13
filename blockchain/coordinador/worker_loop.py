import json
import uuid
import time
import logging
import metrics
import config.settings as settings
from queue_client import queueConnect, publicar_a_pool_manager
from redis_client import getUltimoBlock, gpus_vivas
from storage_client import subirBlock

logger = logging.getLogger("coordinator")


def processPackages(bucket):
    connection, channel = queueConnect()

    while True:
        try:
            txs = []

            metrics.gpus_alive.set(gpus_vivas())
            metrics.update_uptime()

            for _ in range(settings.MAX_TRANSACTIONS_PER_BLOCK):
                method_frame, _, body = channel.basic_get(queue="QueueTransactions")
                if not method_frame:
                    break

                txs.append(json.loads(body))
                channel.basic_ack(method_frame.delivery_tag)

            if txs:
                blockId = str(uuid.uuid4())
                metrics.blocks_created_total.inc()
                last = getUltimoBlock()

                gpus = gpus_vivas()
                difficulty = (
                    settings.DIFFICULTY_LOW if gpus == 0 else settings.DIFFICULTY_HIGH
                )

                block = {
                    "blockId": blockId,
                    "transactions": txs,
                    "prefijo": "0" * difficulty,
                    "baseStringChain": settings.BASE_STRING_CHAIN,
                    "blockchainContent": last["blockchainContent"] if last else "0",
                    "numMaxRandom": settings.MAX_RANDOM,
                }

                subirBlock(bucket, block)
                publicar_a_pool_manager(block, channel)

            connection.process_data_events(time_limit=1)
            time.sleep(settings.PROCESSING_TIME)

        except Exception:
            logger.exception("Error en processPackages")
            connection, channel = queueConnect()

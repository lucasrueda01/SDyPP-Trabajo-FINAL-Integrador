import json
import uuid
import time
import logging
import metrics
import config.settings as settings
from queue_client import queueConnect, publicar_a_pool_manager
from redis_client import getUltimoBlock, gpus_vivas
from storage_client import subirBlock
from redis_client import get_runtime_config

logger = logging.getLogger("coordinator")


def processPackages(bucket):
    connection, channel = queueConnect()

    while True:
        try:
            txs = []

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
                runtime_config = get_runtime_config()

                # -------- DIFFICULTY --------
                if runtime_config["difficulty"] > 0:
                    difficulty = runtime_config["difficulty"]
                else:
                    gpus = gpus_vivas()
                    difficulty = (
                        settings.DIFFICULTY_LOW
                        if gpus == 0
                        else settings.DIFFICULTY_HIGH
                    )

                # -------- ARMADO DEL BLOQUE --------
                block = {
                    "blockId": blockId,
                    "transactions": txs,
                    "prefijo": "0" * difficulty,
                    "baseStringChain": settings.BASE_STRING_CHAIN,
                    "blockchainContent": last["blockchainContent"] if last else "0",
                    "numMaxRandom": runtime_config["max_random"],
                }

                if runtime_config["mining_mode"] and runtime_config["mining_mode"] in [
                    "cooperative",
                    "competitive",
                ]:
                    block["mining_mode"] = runtime_config["mining_mode"]

                if runtime_config["fragment_percent"]:
                    block["fragment_percent"] = runtime_config["fragment_percent"]

                logger.debug(
                    f"Bloque {blockId} creado con {len(txs)} transacciones, dificultad {difficulty}, modo minería: {block.get('mining_mode', 'n/a')}"
                )

                subirBlock(bucket, block)
                publicar_a_pool_manager(block, channel)

            connection.process_data_events(time_limit=1)
            if not txs:
                time.sleep(0.2)  # Evitar loop muy rápido cuando no hay transacciones

            # Bloque resultante hacia pool manager:
            # {
            #  "blockId": id del bloque,
            #  "transactions": [...],
            #  "prefijo": "00",
            #  "baseStringChain": "string",
            #  "blockchainContent": "string",
            #  "mining_mode": "cooperative" o "competitive",
            #  "fragment_percent": 0.5 (opcional, solo para cooperative)
            # }

        except Exception:
            logger.exception("Error en processPackages, reconectando...")
            try:
                connection.close()
            except:
                pass
            connection, channel = queueConnect()

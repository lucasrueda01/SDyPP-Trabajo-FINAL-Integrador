import json
import uuid
import time
import threading
import logging

import metrics
import config.settings as settings
from queue_client import queueConnect, publicar_a_pool_manager
from redis_client import get_redis, getUltimoBlock, gpus_vivas, get_runtime_config
from storage_client import subirBlock

logger = logging.getLogger("coordinator")


def processPackages(bucket):
    redis = get_redis()

    connection, channel = queueConnect()

    MAX_TX = settings.MAX_TRANSACTIONS_PER_BLOCK
    MAX_WAIT_SECONDS = 0.5  # ventana para acumular transacciones

    while True:
        try:
            metrics.update_uptime()

            # 1️⃣ Obtener head actual
            last = getUltimoBlock()
            prev_hash = last["blockchainContent"] if last else "0"

            # 2️⃣ Intentar lock antes de consumir
            lock_key = f"create_lock:{prev_hash}"
            lock_acquired = redis.set(lock_key, "1", nx=True, ex=10)

            if not lock_acquired:
                time.sleep(0.1)
                continue

            # 3️⃣ Consumir con ventana temporal
            txs = []
            start_time = time.time()

            while len(txs) < MAX_TX:
                try:
                    method_frame, _, body = channel.basic_get(queue="QueueTransactions")
                except Exception:
                    logger.warning("Conexión perdida en basic_get. Reconectando...")
                    try:
                        connection.close()
                    except:
                        pass
                    connection, channel = queueConnect()
                    break

                if method_frame:
                    txs.append(json.loads(body))
                    channel.basic_ack(method_frame.delivery_tag)
                else:
                    # No hay mensaje → evaluar ventana temporal
                    if time.time() - start_time > MAX_WAIT_SECONDS:
                        break
                    time.sleep(0.05)

            if not txs:
                time.sleep(0.2)
                continue

            # 4️⃣ Determinar dificultad
            runtime_config = get_runtime_config()

            if runtime_config["difficulty"] > 0:
                difficulty = runtime_config["difficulty"]
            else:
                gpus = gpus_vivas()
                difficulty = (
                    settings.DIFFICULTY_LOW if gpus == 0 else settings.DIFFICULTY_HIGH
                )

            # 5️⃣ Construcción del bloque
            blockId = str(uuid.uuid4())
            metrics.blocks_created_total.inc()

            block = {
                "blockId": blockId,
                "transactions": txs,
                "prefijo": "0" * difficulty,
                "baseStringChain": settings.BASE_STRING_CHAIN,
                "blockchainContent": prev_hash,
                "numMaxRandom": runtime_config["max_random"],
            }

            if runtime_config.get("mining_mode"):
                block["mining_mode"] = runtime_config["mining_mode"]

            if runtime_config.get("fragment_percent"):
                block["fragment_percent"] = runtime_config["fragment_percent"]

            logger.debug(
                f"Bloque {blockId} creado con {len(txs)} transacciones, "
                f"dificultad {difficulty}, modo minería: "
                f"{block.get('mining_mode', 'n/a')}"
            )

            redis.set(f"block:{blockId}:status", "PENDING")
            redis.set(f"block:{blockId}:prev_hash", prev_hash)

            subirBlock(bucket, block)

            # 6️⃣ Publicar a pool manager
            try:
                publicar_a_pool_manager(block)
            except Exception:
                logger.warning("Error publicando bloque. Reintentando...")
                publicar_a_pool_manager(block)

            time.sleep(0.05)

        except Exception:
            logger.exception("Error inesperado en processPackages")
            try:
                connection.close()
            except:
                pass
            time.sleep(2)
            connection, channel = queueConnect()

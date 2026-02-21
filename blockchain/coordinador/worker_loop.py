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
    connection, channel = queueConnect()
    redis = get_redis()

    pending_txs = []
    last_flush_time = time.time()
    FLUSH_INTERVAL = 5  # segundos para forzar bloque chico

    channel.basic_qos(prefetch_count=100)

    def on_transaction(ch, method, properties, body):
        nonlocal pending_txs, last_flush_time

        try:
            metrics.update_uptime()

            tx = json.loads(body)
            pending_txs.append(tx)
            ch.basic_ack(method.delivery_tag)

            now = time.time()

            block_full = len(pending_txs) >= settings.MAX_TRANSACTIONS_PER_BLOCK
            timeout_reached = pending_txs and (now - last_flush_time >= FLUSH_INTERVAL)

            if not block_full and not timeout_reached:
                return

            # Obtener head actual
            last = getUltimoBlock()
            prev_hash = last["blockchainContent"] if last else "0"

            # Lock por head (1 bloque por prev_hash)
            lock_key = f"create_lock:{prev_hash}"

            if not redis.set(lock_key, "1", nx=True, ex=10):
                # Otro coordinador ya está creando bloque
                return

            # Crear bloque
            blockId = str(uuid.uuid4())
            metrics.blocks_created_total.inc()
            runtime_config = get_runtime_config()

            if runtime_config["difficulty"] > 0:
                difficulty = runtime_config["difficulty"]
            else:
                gpus = gpus_vivas()
                difficulty = (
                    settings.DIFFICULTY_LOW if gpus == 0 else settings.DIFFICULTY_HIGH
                )

            block = {
                "blockId": blockId,
                "transactions": pending_txs.copy(),
                "prefijo": "0" * difficulty,
                "baseStringChain": settings.BASE_STRING_CHAIN,
                "blockchainContent": prev_hash,
                "numMaxRandom": runtime_config["max_random"],
            }

            if runtime_config.get("mining_mode") in ["cooperative", "competitive"]:
                block["mining_mode"] = runtime_config["mining_mode"]

            if runtime_config.get("fragment_percent"):
                block["fragment_percent"] = runtime_config["fragment_percent"]

            logger.debug(
                f"Bloque {blockId} creado con {len(pending_txs)} transacciones, "
                f"dificultad {difficulty}, modo minería: "
                f"{block.get('mining_mode', 'n/a')}"
            )

            # Registrar estado
            redis.set(f"block:{blockId}:status", "PENDING")
            redis.set(f"block:{blockId}:prev_hash", prev_hash)

            # Persistir y publicar
            subirBlock(bucket, block)
            publicar_a_pool_manager(block, channel)

            # Limpiar buffer
            pending_txs.clear()
            last_flush_time = now

        except Exception:
            logger.exception("Error procesando transacción en coordinator")

    channel.basic_consume(
        queue="QueueTransactions",
        on_message_callback=on_transaction,
    )

    logger.info(
        "Coordinator consuming transactions (activo-activo con lock por head)..."
    )
    channel.start_consuming()

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
    while True:
        try:
            connection, channel = queueConnect()
            redis = get_redis()

            pending_txs = []
            last_flush_time = time.time()
            FLUSH_INTERVAL = 5  # segundos

            channel.basic_qos(prefetch_count=100)

            def on_transaction(ch, method, properties, body):
                nonlocal pending_txs, last_flush_time

                try:
                    metrics.update_uptime()

                    tx = json.loads(body)
                    pending_txs.append(tx)
                    ch.basic_ack(method.delivery_tag)

                    now = time.time()

                    # 🔹 Condición 1: bloque lleno
                    block_full = len(pending_txs) >= settings.MAX_TRANSACTIONS_PER_BLOCK

                    # 🔹 Condición 2: timeout para bloque chico
                    timeout_reached = pending_txs and (
                        now - last_flush_time >= FLUSH_INTERVAL
                    )

                    if not block_full and not timeout_reached:
                        return

                    # Verificar límite de bloques pendientes
                    last = getUltimoBlock()
                    prev_hash = last["blockchainContent"] if last else "0"
                    pending_key = f"pending:{prev_hash}"

                    max_pending = settings.MAX_PENDING_PER_PREV_HASH
                    current_pending = redis.get(pending_key)
                    current_pending = int(current_pending) if current_pending else 0

                    if current_pending >= max_pending:
                        logger.debug(
                            "Límite de bloques pendientes alcanzado para %s (%d)",
                            prev_hash,
                            current_pending,
                        )
                        return

                    redis.incr(pending_key)

                    blockId = str(uuid.uuid4())
                    metrics.blocks_created_total.inc()
                    runtime_config = get_runtime_config()

                    # Determinar dificultad
                    if runtime_config["difficulty"] > 0:
                        difficulty = runtime_config["difficulty"]
                    else:
                        gpus = gpus_vivas()
                        difficulty = (
                            settings.DIFFICULTY_LOW
                            if gpus == 0
                            else settings.DIFFICULTY_HIGH
                        )

                    # Construcción del bloque
                    block = {
                        "blockId": blockId,
                        "transactions": pending_txs.copy(),
                        "prefijo": "0" * difficulty,
                        "baseStringChain": settings.BASE_STRING_CHAIN,
                        "blockchainContent": prev_hash,
                        "numMaxRandom": runtime_config["max_random"],
                    }

                    if runtime_config.get("mining_mode") in [
                        "cooperative",
                        "competitive",
                    ]:
                        block["mining_mode"] = runtime_config["mining_mode"]

                    if runtime_config.get("fragment_percent"):
                        block["fragment_percent"] = runtime_config["fragment_percent"]

                    logger.debug(
                        f"Bloque {blockId} creado con {len(pending_txs)} transacciones, "
                        f"dificultad {difficulty}, modo minería: "
                        f"{block.get('mining_mode', 'n/a')}"
                    )

                    # Registrar estado en Redis
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

            logger.info("Coordinator consuming transactions...")
            channel.start_consuming()
        except Exception:
            logger.exception("Rabbit perdio conexion, reconectando...")
            time.sleep(5)

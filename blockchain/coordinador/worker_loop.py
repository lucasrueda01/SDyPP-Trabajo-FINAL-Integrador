import json
import uuid
import time
import logging

import metrics
import config.settings as settings
from queue_client import queueConnect, publicar_a_pool_manager
from redis_client import get_redis, getUltimoBlock, gpus_vivas
from storage_client import subirBlock
from redis_client import get_runtime_config

logger = logging.getLogger("coordinator")


def processPackages(bucket):
    connection, channel = queueConnect()
    redis = get_redis()

    while True:
        try:
            metrics.update_uptime()

            # --------------------------------------------------
            # 1️⃣ Verificar si podemos crear un nuevo bloque
            # --------------------------------------------------
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
                time.sleep(0.1)
                continue

            # --------------------------------------------------
            # 2️⃣ Consumir transacciones SOLO si hay slot
            # --------------------------------------------------
            txs = []

            for _ in range(settings.MAX_TRANSACTIONS_PER_BLOCK):
                method_frame, _, body = channel.basic_get(queue="QueueTransactions")
                if not method_frame:
                    break

                txs.append(json.loads(body))
                channel.basic_ack(method_frame.delivery_tag)

            if not txs:
                time.sleep(0.2)
                continue

            # Evitar bloques demasiado pequeños
            if len(txs) < settings.MAX_TRANSACTIONS_PER_BLOCK:
                time.sleep(0.05)

            # --------------------------------------------------
            # 3️⃣ Reservar slot de bloque pendiente
            # --------------------------------------------------
            redis.incr(pending_key)

            blockId = str(uuid.uuid4())
            metrics.blocks_created_total.inc()
            runtime_config = get_runtime_config()

            # --------------------------------------------------
            # 4️⃣ Determinar dificultad
            # --------------------------------------------------
            if runtime_config["difficulty"] > 0:
                difficulty = runtime_config["difficulty"]
            else:
                gpus = gpus_vivas()
                difficulty = (
                    settings.DIFFICULTY_LOW if gpus == 0 else settings.DIFFICULTY_HIGH
                )

            # --------------------------------------------------
            # 5️⃣ Construcción del bloque
            # --------------------------------------------------
            block = {
                "blockId": blockId,
                "transactions": txs,
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
                f"Bloque {blockId} creado con {len(txs)} transacciones, "
                f"dificultad {difficulty}, modo minería: "
                f"{block.get('mining_mode', 'n/a')}"
            )

            # --------------------------------------------------
            # 6️⃣ Registrar estado en Redis
            # --------------------------------------------------
            redis.set(f"block:{blockId}:status", "PENDING")
            redis.set(f"block:{blockId}:prev_hash", prev_hash)

            # --------------------------------------------------
            # 7️⃣ Persistir y publicar
            # --------------------------------------------------
            subirBlock(bucket, block)
            publicar_a_pool_manager(block, channel)

            connection.process_data_events(time_limit=1)

        except Exception:
            logger.exception("Error en processPackages, reconectando...")
            try:
                connection.close()
            except:
                pass
            connection, channel = queueConnect()


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

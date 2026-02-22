import logging
import metrics
import time
from redis_client import (
    get_blockchain_height,
    get_redis,
    existBlock,
    is_block_sealed,
    postBlock,
    release_claim,
    getUltimoBlock,
)
from storage_client import descargarBlock, borrarBlock
from blockchain_service import calculateHash, construirNuevoBloque
from queue_client import encolar

logger = logging.getLogger("coordinator")


def procesar_resultado_worker(data, bucket):
    # Bloque recibido del worker:
    # {
    #     "blockId": blockId,
    #     "workerId": worker_id,
    #     "type": "cpu o gpu",
    #     "processingTime": processing_time,
    #     "hashRate": hashes por segundo,
    #     "hash": hash encontrado,
    #     "intentos": cantidad de intentos realizados
    #     "result": nonce encontrado
    # }

    redisClient = get_redis()
    validation_start = time.time()

    # 0) Validación básica
    if not data or not data.get("result"):
        logger.debug(
            "Resultado inválido recibido del worker %s | Bloque %s",
            data.get("workerId", "unknown"),
            data.get("blockId", "unknown"),
        )
        return {"message": "Resultado invalido"}, 202

    block_id = data["blockId"]
    worker_id = data.get("workerId", "unknown")
    worker_type = data.get("type", "cpu")
    lock_key = None

    status_key = f"block:{block_id}:status"
    claim_key = f"block:{block_id}:claim"

    # 1) Si ya está sellado, ignorar
    if is_block_sealed(block_id):
        metrics.record_task_result(worker_type=worker_type, accepted=False)
        logger.debug(
            "Bloque %s ya cerrado. Recibido del worker %s",
            block_id,
            worker_id,
        )
        return {"message": "Bloque ya cerrado"}, 202

    # 2) Claim exclusivo
    claim_successful = redisClient.set(claim_key, worker_id, nx=True, ex=15)
    if not claim_successful:
        metrics.record_task_result(worker_type=worker_type, accepted=False)
        logger.debug(
            "Bloque %s ya reclamado. Recibido del worker %s",
            block_id,
            worker_id,
        )
        return {"message": "Bloque ya reclamado"}, 202

    try:
        # 3) Descargar bloque temporal
        block = descargarBlock(bucket, block_id)
        if block is None:
            redisClient.set(status_key, "SEALED")
            release_claim(claim_key, worker_id)
            metrics.record_task_result(worker_type=worker_type, accepted=False)
            logger.debug(
                "Bloque %s ya cerrado (no existe temporal). Recibido del worker %s",
                block_id,
                worker_id,
            )
            return {"message": "Bloque ya cerrado"}, 202

        lock_key = f"create_lock:{block['blockchainContent']}"
        # 4) Validar hash
        hash_base = block["baseStringChain"] + block["blockchainContent"]
        hash_calc = calculateHash(data["result"] + hash_base)

        if hash_calc != data["hash"]:
            release_claim(claim_key, worker_id)
            metrics.record_task_result(worker_type=worker_type, accepted=False)
            logger.debug(
                "Bloque %s tiene hash inválido. Recibido del worker %s",
                block_id,
                worker_id,
            )
            return {"message": "Hash invalido"}, 202

        # 5) Verificar que el head no cambió
        prev_actual = getUltimoBlock()
        prev_hash_actual = prev_actual["blockchainContent"] if prev_actual else "0"

        if block["blockchainContent"] != prev_hash_actual:
            metrics.record_block_rejected(stale=True)
            orphan_key = f"block:{block_id}:orphaned"

            was_set = redisClient.set(orphan_key, "1", nx=True)

            if was_set:
                logger.debug(
                    "Reencolando %d txs del bloque huérfano %s",
                    len(block["transactions"]),
                    block_id,
                )
                for tx in block["transactions"]:
                    encolar(tx)

            release_claim(claim_key, worker_id)
            metrics.record_task_result(worker_type=worker_type, accepted=False)
            # Liberar lock porque este bloque ya no es válido
            prev_hash = block["blockchainContent"]
            lock_key = f"create_lock:{prev_hash}"
            redisClient.delete(lock_key)
            logger.debug(
                "Fork detectado para bloque %s. Prev hash cambió.",
                block_id,
            )
            return {"message": "Fork detectado"}, 202

        # 6) Sellar bloque
        redisClient.set(status_key, "SEALED")
        pending_count = len(redisClient.keys("block:*:status"))
        metrics.set_pending_blocks(pending_count)

        # 7) Construir bloque final
        newBlock = construirNuevoBloque(
            block=block,
            prev=prev_actual,
            result_hash=data["hash"],
            nonce=data["result"],
        )

        postBlock(newBlock)
        metrics.set_block_height(get_blockchain_height()["height"])

        # 8) Borrar bloque temporal
        borrarBlock(bucket, block_id)

        logger.debug(
            "Bloque %s aceptado. Ganador: %s. Intentos: %d",
            block_id,
            worker_id,
            data.get("intentos", 0),
        )

        # =========================
        # MÉTRICAS IMPORTANTES
        # =========================

        # ✔ Bloque aceptado
        metrics.record_block_accepted()

        # ✔ Registrar métricas del worker
        metrics.record_task_result(
            worker_type=worker_type,
            accepted=True,
            processing_time=data.get("processingTime"),
            attempts=data.get("intentos"),
            hash_rate_value=data.get("hashRate"),
        )

        # ✔ Latencia total del bloque
        metrics.record_block_latency(data["latency"])

        # ✔ Tiempo de validación coordinador
        validation_time = time.time() - validation_start
        metrics.record_validation_time(validation_time)

        return {"message": "Bloque aceptado"}, 201

    except Exception:
        release_claim(claim_key, worker_id)
        metrics.errors_total.inc()
        logger.exception(
            "Error procesando resultado del worker %s para bloque %s",
            worker_id,
            block_id,
        )
        return {"message": "Error interno"}, 500
    finally:
        if lock_key:
            redisClient.delete(lock_key)

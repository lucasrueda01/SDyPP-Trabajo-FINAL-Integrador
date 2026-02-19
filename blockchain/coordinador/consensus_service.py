import logging
import metrics
from redis_client import (
    get_redis,
    existBlock,
    is_block_sealed,
    postBlock,
    release_claim,
    getUltimoBlock,
    release_pending_slot
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
    #     "result": nonce encontrado
    # }

    redisClient = get_redis()

    if not data or not data.get("result"):
        logger.debug(
            "Resultado inválido recibido del worker %s:  | Bloque %s",
            data.get("workerId", "unknown"),
            data.get("blockId", "unknown"),
        )
        return {"message": "Resultado invalido"}, 202

    block_id = data["blockId"]
    worker_id = data.get("workerId", "unknown")
    worker_type = data.get("type", "cpu")

    status_key = f"block:{block_id}:status"
    claim_key = f"block:{block_id}:claim"

    # -----------------------
    # 1) Idempotencia fuerte
    # -----------------------
    if is_block_sealed(block_id):
        metrics.record_task_result(worker_type=worker_type, accepted=False)
        metrics.blocks_rejected_total.inc()
        release_pending_slot(redisClient, block_id)
        logger.debug(
            "Bloque %s ya cerrado. Recibido del worker %s", block_id, worker_id
        )
        return {"message": "Bloque ya cerrado"}, 202

    # -----------------------
    # 2) Claim exclusivo
    # -----------------------
    if not redisClient.set(claim_key, worker_id, nx=True, ex=15):
        metrics.blocks_rejected_total.inc()
        metrics.record_task_result(worker_type=worker_type, accepted=False)
        logger.debug(
            "Bloque %s ya reclamado. Recibido del worker %s", block_id, worker_id
        )
        return {"message": "Bloque ya reclamado"}, 202

    try:
        # -----------------------
        # 3) Descargar bloque
        # -----------------------
        block = descargarBlock(bucket, block_id)
        if block is None:
            redisClient.set(status_key, "SEALED")
            release_claim(claim_key, worker_id)
            release_pending_slot(redisClient, block_id)
            metrics.blocks_rejected_total.inc()
            metrics.record_task_result(worker_type=worker_type, accepted=False)
            logger.debug(
                "Bloque %s ya cerrado. Recibido del worker %s", block_id, worker_id
            )
            return {"message": "Bloque ya cerrado"}, 202

        # -----------------------
        # 4) Validar hash
        # -----------------------
        hash_base = block["baseStringChain"] + block["blockchainContent"]
        hash_calc = calculateHash(data["result"] + hash_base)

        if hash_calc != data["hash"]:
            release_claim(claim_key, worker_id)
            release_pending_slot(redisClient, block_id)
            metrics.blocks_rejected_total.inc()
            metrics.record_task_result(worker_type=worker_type, accepted=False)
            logger.debug(
                "Bloque %s tiene hash inválido. Recibido del worker %s",
                block_id,
                worker_id,
            )
            return {"message": "Hash invalido"}, 202

        # -----------------------
        # 5) Verificar que el prev siga siendo el actual
        # -----------------------
        prev_actual = getUltimoBlock()

        prev_hash_actual = prev_actual["blockchainContent"] if prev_actual else "0"

        if block["blockchainContent"] != prev_hash_actual:
            # Intentamos marcar el bloque como huérfano solo una vez
            orphan_key = f"block:{block_id}:orphaned"

            was_set = redisClient.set(orphan_key, "1", nx=True)

            if was_set:
                logger.debug(
                    "Reencolando tx del bloque huérfano %s",
                    block_id,
                )

                for tx in block["transactions"]:
                    encolar(tx)  # Usá tu función real de encolado

            release_pending_slot(redisClient, block_id)


            release_claim(claim_key, worker_id)
            metrics.blocks_rejected_total.inc()
            metrics.record_task_result(worker_type=worker_type, accepted=False)

            logger.debug(
                "Fork detectado para bloque %s. Prev hash cambió.",
                block_id,
            )
            return {"message": "Fork detectado"}, 202

        # -----------------------
        # 6) Sellar bloque
        # -----------------------
        redisClient.set(status_key, "SEALED")
        release_pending_slot(redisClient, block_id)

        # -----------------------
        # 7) Construir bloque final
        # -----------------------
        newBlock = construirNuevoBloque(
            block=block,
            prev=prev_actual,
            result_hash=data["hash"],
            nonce=data["result"],
        )

        postBlock(newBlock)

        # -----------------------
        # 8) Borrar temporal
        # -----------------------
        borrarBlock(bucket, block_id)

        logger.debug("Bloque %s aceptado. Ganador: %s", block_id, worker_id)

        metrics.blocks_accepted_total.inc()
        metrics.record_task_result(
            worker_type=worker_type,
            accepted=True,
            processing_time=data.get("processingTime"),
            hash_rate=data.get("hashRate"),
        )

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

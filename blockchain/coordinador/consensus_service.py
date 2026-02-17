import logging
import metrics
from redis_client import (
    get_redis,
    existBlock,
    postBlock,
    release_claim,
    getUltimoBlock,
)
from storage_client import descargarBlock, borrarBlock
from blockchain_service import calculateHash, construirNuevoBloque

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
    status = redisClient.get(status_key)
    if status and status.decode() == "SEALED":
        metrics.record_task_result(worker_type=worker_type, accepted=False)
        metrics.blocks_rejected_total.inc()
        logger.debug("Bloque %s ya cerrado. Worker: %s", block_id, worker_id)
        return {"message": "Bloque ya cerrado"}, 202

    # -----------------------
    # 2) Claim exclusivo
    # -----------------------
    if not redisClient.set(claim_key, worker_id, nx=True, ex=15):
        metrics.blocks_rejected_total.inc()
        metrics.record_task_result(worker_type=worker_type, accepted=False)
        logger.debug("Bloque %s ya reclamado. Worker: %s", block_id, worker_id)
        return {"message": "Bloque ya reclamado"}, 202

    try:
        # -----------------------
        # 3) Descargar bloque
        # -----------------------
        block = descargarBlock(bucket, block_id)
        if block is None:
            redisClient.set(status_key, "SEALED")
            release_claim(claim_key, worker_id)
            metrics.blocks_rejected_total.inc()
            metrics.record_task_result(worker_type=worker_type, accepted=False)
            logger.debug("Bloque %s ya cerrado. Worker: %s", block_id, worker_id)
            return {"message": "Bloque ya cerrado"}, 202

        # -----------------------
        # 4) Validar hash
        # -----------------------
        hash_base = block["baseStringChain"] + block["blockchainContent"]
        hash_calc = calculateHash(data["result"] + hash_base)

        if hash_calc != data["hash"]:
            release_claim(claim_key, worker_id)
            metrics.blocks_rejected_total.inc()
            metrics.record_task_result(worker_type=worker_type, accepted=False)
            logger.debug(
                "Bloque %s tiene hash inválido. Worker: %s", block_id, worker_id
            )
            return {"message": "Hash invalido"}, 202

        # -----------------------
        # 5) Verificar existencia
        # -----------------------
        if existBlock(block_id):
            redisClient.set(status_key, "SEALED")
            release_claim(claim_key, worker_id)
            metrics.blocks_rejected_total.inc()
            metrics.record_task_result(worker_type=worker_type, accepted=False)
            logger.debug("Bloque %s ya existe. Worker: %s", block_id, worker_id)
            return {"message": "Bloque ya existe"}, 202

        # -----------------------
        # 6) Sellar bloque
        # -----------------------
        redisClient.set(status_key, "SEALED")

        # -----------------------
        # 7) Construir bloque final
        # -----------------------
        prev = getUltimoBlock()

        newBlock = construirNuevoBloque(
            block=block,
            prev=prev,
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

import json
import logging
import config.settings as settings
from pool_manager.fragmenter import fragmentar
from pool_manager.rabbitmq import safe_publish

logger = logging.getLogger("pool-manager")


def dispatch_to_workers(block, alive_workers, channel):
    block_id = block["blockId"]

    if not alive_workers:
        logger.warning("No hay workers vivos para %s", block_id)
        return False

    if not settings.COOPERATIVE_MINING:
        logger.info("Despachando %s en COMPETITIVO", block_id)
        
        logger.info("Rango nonce completo: %d - %d. Workers vivos: %d", block["nonce_start"], block["nonce_end"], len(alive_workers))
        safe_publish(
            channel,
            "blocks_competitive",
            "",
            json.dumps(block),
        )
        return True

    logger.info("Despachando %s en COOPERATIVO", block_id)

    gpu_workers = [w for w in alive_workers if w["type"] == "gpu"]
    cpu_workers = [w for w in alive_workers if w["type"] == "cpu"]

    gpu_payloads, cpu_payloads = fragmentar(
        block,
        len(gpu_workers),
        len(cpu_workers),
    )

    for i, payload in enumerate(gpu_payloads):
        start = payload["nonce_start"]
        end = payload["nonce_end"]

        logger.info(
            "Asignando GPU %d/%d -> nonce %d-%d",
            i + 1,
            len(gpu_payloads),
            start,
            end,
        )

        safe_publish(
            channel,
            "blocks_cooperative",
            "blocks.gpu",
            json.dumps(payload),
        )

    for i, payload in enumerate(cpu_payloads):
        start = payload["nonce_start"]
        end = payload["nonce_end"]

        logger.info(
            "Asignando CPU %d/%d -> nonce %d-%d",
            i + 1,
            len(cpu_payloads),
            start,
            end,
        )

        safe_publish(
            channel,
            "blocks_cooperative",
            "blocks.cpu",
            json.dumps(payload),
        )

    return True

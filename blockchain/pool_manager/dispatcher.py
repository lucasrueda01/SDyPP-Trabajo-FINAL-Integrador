import json
import logging
import config.settings as settings
from rabbitmq import safe_publish

logger = logging.getLogger("pool-manager")


def fragmentar(block, total_workers):
    nonce_start = block.get("nonce_start", 0)
    nonce_end = block.get(
        "nonce_end",
        block.get("numMaxRandom", settings.MAX_RANDOM),
    )
    total_space = nonce_end - nonce_start + 1
    if total_space <= 0 or total_workers <= 0:
        return []
    fragment_percent = block.get(
        "fragment_percent",
        settings.FRAGMENT_PERCENT,
    )
    fragment_size = int(total_space * fragment_percent)

    if fragment_size <= 0:
        fragment_size = 1
    payloads = []
    cursor = nonce_start

    while cursor <= nonce_end:
        end = min(cursor + fragment_size - 1, nonce_end)

        payload = {
            **block,
            "nonce_start": cursor,
            "nonce_end": end,
        }

        payloads.append(payload)
        cursor = end + 1

    return payloads


def dispatch_to_workers(block, alive_workers, channel):
    block_id = block["blockId"]

    if not alive_workers:
        logger.warning("No hay workers vivos para %s", block_id)
        return False

    # Determinar modo
    if "mining_mode" in block:
        mining_mode = block["mining_mode"]
    else:
        mining_mode = "cooperative" if settings.COOPERATIVE_MINING else "competitive"

    # MODO COMPETITIVO
    if mining_mode == "competitive":
        logger.info("Despachando %s en COMPETITIVO", block_id)

        safe_publish(
            channel,
            "blocks_competitive",
            "",
            json.dumps(block),
        )

        return True

    # MODO COOPERATIVO
    logger.info("Despachando %s en COOPERATIVO", block_id)

    # total_workers ahora es simple
    total_workers = len(alive_workers)

    payloads = fragmentar(block, total_workers)

    for i, payload in enumerate(payloads):
        logger.info(
            "Asignando fragmento %d/%d -> nonce %d-%d",
            i + 1,
            len(payloads),
            payload["nonce_start"],
            payload["nonce_end"],
        )

        safe_publish(
            channel,
            "blocks_cooperative",
            "",
            json.dumps(payload),
        )

    return True

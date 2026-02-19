import redis
import json
import logging
import config.settings as settings

logger = logging.getLogger("coordinator")

redisClient = None


def redisConnect():
    global redisClient
    redisClient = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
    )
    redisClient.ping()
    logger.info("Conectado a Redis")
    return redisClient


def get_redis():
    global redisClient
    try:
        redisClient.ping()
    except Exception:
        logger.warning("Redis reconectando...")
        redisConnect()
    return redisClient


def getUltimoBlock():
    client = get_redis()
    raw = client.lindex("blockchain", 0)
    return json.loads(raw) if raw else None


def existBlock(block_id):
    return get_redis().sismember("block_ids", block_id)


def postBlock(block):
    client = get_redis()
    pipe = client.pipeline()
    pipe.lpush("blockchain", json.dumps(block))
    pipe.sadd("block_ids", block["blockId"])
    pipe.execute()


def release_claim(claim_key, worker_id):
    client = get_redis()
    owner = client.get(claim_key)
    if owner and owner.decode() == worker_id:
        client.delete(claim_key)


def is_block_sealed(block_id):
    redisClient = get_redis()

    status_key = f"block:{block_id}:status"

    status = redisClient.get(status_key)
    if status and status.decode() == "SEALED":
        return True

    if existBlock(block_id):
        return True

    return False


def gpus_vivas():
    client = get_redis()
    count = 0
    for key in client.scan_iter("worker:*"):
        data = client.hgetall(key)
        if data.get(b"type") == b"gpu":
            count += 1
    return count


def get_blockchain():
    client = get_redis()
    chain = client.lrange("blockchain", 0, -1)
    return [json.loads(b.decode("utf-8")) for b in chain]


def get_blockchain_height():
    client = get_redis()
    height = client.llen("blockchain")
    return {"height": height}


def get_runtime_config():
    client = get_redis()
    config = client.hgetall("system:config")

    return {
        "fragment_percent": float(
            config.get(b"fragment_percent", settings.FRAGMENT_PERCENT)
        ),
        "max_random": int(config.get(b"max_random", settings.MAX_RANDOM)),
        "difficulty": int(config.get(b"difficulty", 0)),
        "mining_mode": config.get(b"mining_mode", b"cooperative").decode(),
    }


def update_runtime_config(new_config):
    client = get_redis()
    pipe = client.pipeline()
    for key, value in new_config.items():
        pipe.hset("system:config", key, value)
    pipe.execute()
    logger.debug("Runtime config actualizada: %s", new_config)


def release_pending_slot(redisClient, block_id):
    flag_key = f"block:{block_id}:slot_released"

    was_set = redisClient.set(flag_key, "1", nx=True)

    if was_set:
        prev_hash = redisClient.get(f"block:{block_id}:prev_hash")
        if prev_hash:
            redisClient.decr(f"pending:{prev_hash.decode()}")


def reset_blockchain_state():
    redis_client = get_redis()
    deleted = 0
    patterns = ["block:*", "pending:*", "blockchain:*"]
    for pattern in patterns:
        for key in redis_client.scan_iter(pattern):
            redis_client.delete(key)
            deleted += 1
    return deleted
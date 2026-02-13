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


def gpus_vivas():
    client = get_redis()
    count = 0
    for key in client.scan_iter("worker:*"):
        data = client.hgetall(key)
        if data.get(b"type") == b"gpu":
            count += 1
    return count

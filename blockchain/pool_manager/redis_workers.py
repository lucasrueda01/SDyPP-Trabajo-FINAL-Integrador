import json
import logging
import redis
import config.settings as settings

logger = logging.getLogger("pool-manager")


def redis_connect():
    client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=True,
    )
    client.ping()
    logger.info("Conectado a Redis")
    return client


def register_worker(redis_client, wid, data, ip):

    worker_data = {
        "id": wid,
        "type": data["type"],
        "capacity": data.get("capacity", 5),
        "ip": ip,
    }

    redis_client.set(
        f"worker:{wid}",
        json.dumps(worker_data),
        ex=settings.HEARTBEAT_TTL + 5,
    )
    
    logger.info(
        "Worker registrado: id=%s type=%s ip=%s",
        wid,
        worker_data["type"],
        ip,
    )

    return worker_data


def heartbeat(redis_client, wid):
    key = f"worker:{wid}"
    if not redis_client.get(key):
        return False
    redis_client.expire(key, settings.HEARTBEAT_TTL + 5)
    logger.debug("Heartbeat recibido de %s", wid)
    return True


def get_alive_workers(redis_client):
    alive = []
    total_capacity = 0

    for key in redis_client.scan_iter("worker:*"):
        raw = redis_client.get(key)
        if not raw:
            continue

        w = json.loads(raw)
        alive.append(w)
        total_capacity += int(w["capacity"])

    return alive, total_capacity

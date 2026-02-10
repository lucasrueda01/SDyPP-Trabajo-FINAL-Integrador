import json
import logging
import time
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

    created = redis_client.set(
        f"worker:{wid}",
        json.dumps(worker_data),
        ex=settings.HEARTBEAT_TTL,
        nx=True,
    )

    if created:
        logger.info(
            "Worker registrado: id=%s type=%s ip=%s",
            wid,
            worker_data["type"],
            ip,
        )
    else:
        logger.debug(
            "Worker %s ya registrado, no se recrea",
            wid,
        )

    return worker_data


def heartbeat(redis_client, wid):
    key = f"worker:{wid}"
    raw = redis_client.get(key)
    if not raw:
        logger.warning("Heartbeat recibido de worker no registrado: %s", wid)
        return False
    data = json.loads(raw)

    data["last_heartbeat"] = time.time()
    redis_client.set(
        key,
        json.dumps(data),
        ex=settings.HEARTBEAT_TTL,
    )

    logger.debug(
        "Heartbeat recibido de worker %s, TTL renovado y last_heartbeat actualizado",
        wid,
    )
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

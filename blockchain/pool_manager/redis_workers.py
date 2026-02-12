import json
import logging
import time
import redis
import config.settings as settings

logger = logging.getLogger("pool-manager")


def redis_connect():
    client = redis.Redis(
        host=settings.REDIS_HOST,
        port=int(settings.REDIS_PORT),
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        health_check_interval=30,
        retry_on_timeout=True,
    )
    client.ping()
    logger.info("Conectado a Redis")
    return client


def get_alive_workers(redis_client):
    alive = []
    total_capacity = 0

    for key in redis_client.scan_iter("worker:*"):
        w = redis_client.hgetall(key)
        if not w:
            continue

        # normalizar tipos
        w["capacity"] = int(w.get("capacity", 0))
        if "last_seen" in w:
            w["last_seen"] = float(w["last_seen"])

        alive.append(w)
        total_capacity += w["capacity"]

    return alive, total_capacity

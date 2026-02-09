import threading
from flask import Flask, request, jsonify
import config.settings as settings
from redis_workers import redis_connect, register_worker, heartbeat
from consumers import start_pool_consumer, start_dlq_consumer
import logging
import sys
import time
import metrics

app = Flask(__name__)
redis_client = redis_connect()

LOG_LEVEL = logging.DEBUG


logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("pool-manager")

logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("redis").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger.info("Pool Manager iniciando")


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    wid = data["id"]
    register_worker(redis_client, wid, data, request.remote_addr)
    metrics.pool_worker_registrations.inc()
    return jsonify({"status": "ok"})


@app.route("/heartbeat", methods=["POST"])
def hb():
    metrics.update_uptime()
    data = request.get_json()
    wid = data["id"]

    ok = heartbeat(redis_client, wid)

    if not ok:
        logger.warning(
            "Heartbeat de %s sin registro previo, recreando estado",
            wid,
        )
        return jsonify({"error": "worker not registered"}), 404
    metrics.worker_heartbeats_total.inc()

    return jsonify({"status": "ok"})


metrics.start_metrics_server(8000)
threading.Thread(target=start_pool_consumer, args=(redis_client,), daemon=True).start()
threading.Thread(target=start_dlq_consumer, args=(redis_client,), daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.POOL_MANAGER_PORT)

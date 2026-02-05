import threading
from flask import Flask, request, jsonify
import config.settings as settings
from pool_manager.redis_workers import redis_connect, register_worker, heartbeat
from pool_manager.consumers import start_pool_consumer, start_dlq_consumer
import logging
import sys

app = Flask(__name__)
redis_client = redis_connect()

LOG_LEVEL = logging.DEBUG


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
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
    return jsonify({"status": "ok"})


@app.route("/heartbeat", methods=["POST"])
def hb():
    data = request.get_json()
    ok = heartbeat(redis_client, data["id"])

    if not ok:
        return jsonify({"error": "not registered"}), 404

    return jsonify({"status": "ok"})




threading.Thread(target=start_pool_consumer, args=(redis_client,), daemon=True).start()
threading.Thread(target=start_dlq_consumer, args=(redis_client,), daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.POOL_MANAGER_PORT)

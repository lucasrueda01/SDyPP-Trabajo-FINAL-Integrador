import logging
import sys
from flask import Flask, jsonify, request
from prometheus_client import start_http_server
import metrics
import threading
import config.settings as settings
from consensus_service import procesar_resultado_worker

from queue_client import encolar
from redis_client import redisConnect
from storage_client import bucketConnect
from blockchain_service import validarTransaction
from worker_loop import processPackages

start_http_server(8000)

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("redis").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger("coordinator")

# Inicializaciones
redisConnect()
bucket = bucketConnect(settings.BUCKET_NAME)

# Background thread
threading.Thread(target=processPackages, args=(bucket,), daemon=True).start()


@app.route("/transaction", methods=["POST"])
def addTransaction():
    tx = request.json

    if not validarTransaction(tx):
        return "Transacción inválida", 400

    metrics.transactions_total.inc()
    encolar(tx)
    return "Transacción aceptada", 202


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "OK"})


@app.route("/solved_task", methods=["POST"])
def receive_solved_task():
    data = request.get_json()
    response, status = procesar_resultado_worker(data, bucket)
    return jsonify(response), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.COORDINADOR_PORT)

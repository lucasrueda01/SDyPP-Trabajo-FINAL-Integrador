import logging
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from prometheus_client import start_http_server
from redis_client import (
    get_blockchain,
    get_blockchain_height,
    get_redis,
    reset_blockchain_state,
    update_runtime_config,
)
import metrics
import threading

import config.settings as settings
from redis_client import get_runtime_config
from consensus_service import procesar_resultado_worker
from queue_client import encolar, init_publisher
from redis_client import redisConnect
from storage_client import bucketConnect
from blockchain_service import validarTransaction
from worker_loop import processPackages
from flask import send_from_directory

start_http_server(8000)

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger("redis").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)


logger = logging.getLogger("coordinator")

# Inicializaciones
redisConnect()
init_publisher()
bucket = bucketConnect(settings.BUCKET_NAME)

# Background thread
threading.Thread(target=processPackages, args=(bucket,), daemon=True).start()


@app.route("/transaction", methods=["POST"])
def addTransaction():
    tx = request.json

    if not validarTransaction(tx):
        logger.warning(f"TX inválida recibida: {tx}")
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


@app.route("/blockchain", methods=["GET"])
def get_bc():
    return jsonify(get_blockchain())


@app.route("/blockchain/height", methods=["GET"])
def bc_height():
    return jsonify(get_blockchain_height())


@app.route("/config", methods=["POST"])
def update_config():
    data = request.json
    update_runtime_config(data)
    return jsonify({"status": "updated"})


@app.route("/config", methods=["GET"])
def get_config():
    return jsonify(get_runtime_config())


@app.route("/admin/reset-blockchain", methods=["POST"])
def reset_blockchain():
    try:
        deleted = reset_blockchain_state()
        return jsonify({"status": "Blockchain reset", "keys_deleted": deleted}), 200

    except Exception as e:
        logger.exception("Error reseteando blockchain")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def serve_frontend():
    return send_from_directory("static", "index.html")
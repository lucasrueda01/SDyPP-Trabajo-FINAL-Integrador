import time
from prometheus_client import Counter, Gauge, start_http_server

_started = False
_start_time = time.time()


def start_metrics_server(port=8000):
    global _started
    if not _started:
        start_http_server(port)
        _started = True


hashes_total = Counter(
    "worker_hashes_total", "Total de hashes calculados", ["worker_type"]
)

blocks_processed_total = Counter(
    "worker_blocks_processed_total", "Total de bloques procesados", ["worker_type"]
)

blocks_solved_total = Counter(
    "worker_blocks_solved_total", "Total de bloques resueltos", ["worker_type"]
)

worker_errors_total = Counter(
    "worker_errors_total", "Errores del worker", ["worker_type"]
)

hash_rate_gauge = Gauge("worker_hash_rate", "Hash rate actual", ["worker_type"])

worker_active = Gauge("worker_active", "Worker activo (1=activo)", ["worker_type"])

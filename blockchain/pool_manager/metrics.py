# metrics.py
from prometheus_client import Counter, Gauge, start_http_server
import time

# =====================
# Inicialización
# =====================
_start_time = time.time()
_metrics_started = False


def start_metrics_server(port=8000):
    global _metrics_started
    if not _metrics_started:
        start_http_server(port)
        _metrics_started = True


# =====================
# Counters (eventos)
# =====================
blocks_received_total = Counter(
    "pool_blocks_received_total",
    "Bloques recibidos desde RabbitMQ"
)

blocks_dispatched_total = Counter(
    "pool_blocks_dispatched_total",
    "Bloques despachados correctamente"
)

blocks_failed_total = Counter(
    "pool_blocks_failed_total",
    "Bloques que fallaron en el despacho"
)

blocks_dlq_total = Counter(
    "pool_blocks_dlq_total",
    "Bloques enviados a DLQ"
)

worker_registrations_total = Counter(
    "pool_worker_registrations_total",
    "Workers registrados en el pool"
)

worker_heartbeats_total = Counter(
    "pool_worker_heartbeats_total",
    "Heartbeats recibidos de workers"
)

# =====================
# Gauges (estado)
# =====================
workers_alive = Gauge(
    "pool_workers_alive",
    "Cantidad total de workers vivos"
)

workers_cpu = Gauge(
    "pool_workers_cpu",
    "Cantidad de workers CPU vivos"
)

workers_gpu = Gauge(
    "pool_workers_gpu",
    "Cantidad de workers GPU vivos"
)

uptime_seconds = Gauge(
    "pool_uptime_seconds",
    "Uptime del pool manager en segundos"
)


def update_uptime():
    uptime_seconds.set(time.time() - _start_time)

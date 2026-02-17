import time
from prometheus_client import Counter, Gauge

# =====================
# Lifecycle
# =====================
_start_time = time.time()

uptime_seconds = Gauge(
    "coordinator_uptime_seconds",
    "Uptime del coordinador en segundos",
)

# =====================
# Transactions / blocks
# =====================
transactions_total = Counter(
    "coordinator_transactions_total",
    "Total de transacciones recibidas",
)

blocks_created_total = Counter(
    "coordinator_blocks_created_total",
    "Total de bloques creados",
)

blocks_accepted_total = Counter(
    "coordinator_blocks_accepted_total",
    "Total de bloques aceptados",
)

blocks_rejected_total = Counter(
    "coordinator_blocks_rejected_total",
    "Total de bloques rechazados",
)

errors_total = Counter(
    "coordinator_errors_total",
    "Errores internos del coordinador",
)

# =====================
# Results from workers (CPU / GPU)
# =====================
tasks_total = Counter(
    "coordinator_tasks_total",
    "Resultados recibidos de workers",
    ["worker_type"],
)

tasks_accepted = Counter(
    "coordinator_tasks_accepted_total",
    "Bloques aceptados por tipo de worker",
    ["worker_type"],
)

tasks_rejected = Counter(
    "coordinator_tasks_rejected_total",
    "Bloques rechazados por tipo de worker",
    ["worker_type"],
)

processing_time_avg_ms = Gauge(
    "coordinator_processing_time_avg_ms",
    "Tiempo promedio de procesamiento por bloque",
    ["worker_type"],
)

hash_rate_avg = Gauge(
    "coordinator_hash_rate_avg",
    "Hash rate promedio por tipo de worker",
    ["worker_type"],
)

hash_rate_last = Gauge(
    "coordinator_hash_rate_last",
    "Último hash rate reportado",
    ["worker_type"],
)

# =====================
# In-memory accumulators
# =====================
_stats = {
    "cpu": {
        "accepted": 0,
        "total_processing_time": 0.0,
        "total_hash_rate": 0.0,
    },
    "gpu": {
        "accepted": 0,
        "total_processing_time": 0.0,
        "total_hash_rate": 0.0,
    },
}


# =====================
# Helper functions
# =====================
def update_uptime():
    uptime_seconds.set(time.time() - _start_time)


def record_task_result(worker_type, accepted, processing_time=None, hash_rate=None):
    if worker_type not in _stats:
        worker_type = "cpu"

    tasks_total.labels(worker_type=worker_type).inc()

    if accepted:
        tasks_accepted.labels(worker_type=worker_type).inc()

        _stats[worker_type]["accepted"] += 1
        _stats[worker_type]["total_processing_time"] += processing_time or 0.0
        _stats[worker_type]["total_hash_rate"] += hash_rate or 0.0

        count = _stats[worker_type]["accepted"]

        processing_time_avg_ms.labels(worker_type=worker_type).set(
            _stats[worker_type]["total_processing_time"] / count
        )

        hash_rate_avg.labels(worker_type=worker_type).set(
            _stats[worker_type]["total_hash_rate"] / count
        )

        if hash_rate is not None:
            hash_rate_last.labels(worker_type=worker_type).set(hash_rate)

    else:
        tasks_rejected.labels(worker_type=worker_type).inc()

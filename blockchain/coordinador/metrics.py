import time
from prometheus_client import Counter, Gauge, Histogram

# =====================
# Lifecycle
# =====================

_start_time = time.time()

uptime_seconds = Gauge(
    "coordinator_uptime_seconds",
    "Uptime del coordinador en segundos",
)

# =====================
# Blockchain state
# =====================

block_height = Gauge(
    "coordinator_block_height",
    "Altura actual de la blockchain",
)

pending_blocks = Gauge(
    "coordinator_pending_blocks",
    "Cantidad de bloques en estado PENDING",
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

stale_blocks_total = Counter(
    "coordinator_stale_blocks_total",
    "Bloques rechazados por hash previo desactualizado",
)

errors_total = Counter(
    "coordinator_errors_total",
    "Errores internos del coordinador",
)

# =====================
# Worker results
# =====================

tasks_total = Counter(
    "coordinator_tasks_total",
    "Resultados recibidos de workers",
    ["worker_type"],
)

tasks_accepted_total = Counter(
    "coordinator_tasks_accepted_total",
    "Bloques aceptados por tipo de worker",
    ["worker_type"],
)

tasks_rejected_total = Counter(
    "coordinator_tasks_rejected_total",
    "Bloques rechazados por tipo de worker",
    ["worker_type"],
)

# =====================
# Mining metrics (CRITICAL)
# =====================

# Intentos necesarios para encontrar el nonce
block_attempts = Histogram(
    "coordinator_block_attempts",
    "Intentos necesarios para minar un bloque",
    ["worker_type"],
    buckets=(1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9),
)

# Tiempo de minería reportado por el worker
mining_time_seconds = Histogram(
    "coordinator_mining_time_seconds",
    "Tiempo de minería reportado por el worker (segundos)",
    ["worker_type"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 60, 120),
)

# Hash rate reportado (último valor)
hash_rate = Gauge(
    "coordinator_hash_rate",
    "Hash rate reportado por el worker",
    ["worker_type"],
)

# =====================
# Coordinator performance
# =====================

# Latencia total desde creación hasta confirmación
block_latency_seconds = Histogram(
    "coordinator_block_latency_seconds",
    "Tiempo total desde creación del bloque hasta confirmación",
    buckets=(0.5, 1, 2, 5, 10, 20, 60, 120, 300),
)

# Tiempo interno de validación en el coordinador
validation_time_seconds = Histogram(
    "coordinator_validation_time_seconds",
    "Tiempo que tarda el coordinador en validar y confirmar un bloque",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2),
)

# =====================
# Helper functions
# =====================


def update_uptime():
    uptime_seconds.set(time.time() - _start_time)


def set_block_height(height: int):
    block_height.set(height)


def set_pending_blocks(count: int):
    pending_blocks.set(count)


def record_block_created():
    blocks_created_total.inc()


def record_block_accepted():
    blocks_accepted_total.inc()


def record_block_rejected(stale=False):
    blocks_rejected_total.inc()
    if stale:
        stale_blocks_total.inc()


def record_task_result(
    worker_type, accepted, processing_time=None, attempts=None, hash_rate_value=None
):
    """
    Registra resultado enviado por un worker.
    """

    if worker_type not in ("cpu", "gpu"):
        worker_type = "cpu"

    tasks_total.labels(worker_type=worker_type).inc()

    if accepted:
        tasks_accepted_total.labels(worker_type=worker_type).inc()

        if processing_time is not None:
            mining_time_seconds.labels(worker_type=worker_type).observe(processing_time)

        if attempts is not None:
            block_attempts.labels(worker_type=worker_type).observe(float(attempts))

        if hash_rate_value is not None:
            hash_rate.labels(worker_type=worker_type).set(hash_rate_value)

    else:
        tasks_rejected_total.labels(worker_type=worker_type).inc()


def record_block_latency(seconds):
    block_latency_seconds.observe(seconds)


def record_validation_time(seconds):
    validation_time_seconds.observe(seconds)

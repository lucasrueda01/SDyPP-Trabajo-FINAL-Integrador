from prometheus_client import Counter, Gauge, start_http_server
import time

_started = False
_start_time = time.time()


def start_metrics_server(port=8000):
    global _started
    if not _started:
        start_http_server(port)
        _started = True


# -------- Counters --------
reconciliations_total = Counter(
    "cpu_scaler_reconciliations_total",
    "Cantidad de ciclos de reconciliación ejecutados"
)

scale_up_total = Counter(
    "cpu_scaler_scale_up_total",
    "Cantidad de veces que se escalo hacia arriba"
)

scale_down_total = Counter(
    "cpu_scaler_scale_down_total",
    "Cantidad de veces que se escalo hacia abajo"
)

vm_created_total = Counter(
    "cpu_scaler_vm_created_total",
    "VMs CPU creadas dinamicamente"
)

vm_deleted_total = Counter(
    "cpu_scaler_vm_deleted_total",
    "VMs CPU eliminadas dinamicamente"
)

errors_total = Counter(
    "cpu_scaler_errors_total",
    "Errores del cpu-scaler"
)

# -------- Gauges --------
dynamic_cpu_workers = Gauge(
    "cpu_scaler_workers_cpu_dynamic",
    "Cantidad de workers CPU dinamicos activos"
)

gpus_missing = Gauge(
    "cpu_scaler_gpus_missing",
    "Cantidad de GPUs faltantes detectadas"
)

target_cpu_workers = Gauge(
    "cpu_scaler_target_cpu_workers",
    "Cantidad objetivo de workers CPU"
)

uptime_seconds = Gauge(
    "pool_uptime_seconds",
    "Uptime del pool manager en segundos"
)

total_workers_cpu = Gauge(
    "workers_cpu_total",
    "Cantidad total de workers CPU vivos"
)

total_workers_gpu = Gauge(
    "workers_gpu_total",
    "Cantidad total de workers GPU vivos"
)

total_workers = Gauge(
    "workers_total",
    "Cantidad total de workers vivos"
)

worker_info = Gauge(
    "worker_info",
    "Informacion de workers vivos segun Redis",
    ["id", "type", "ip", "capacity"],
)

def update_uptime():
    uptime_seconds.set(time.time() - _start_time)
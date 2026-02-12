import os
import time
import json
import logging
import redis

import config.settings as settings
from google.cloud import compute_v1
import metrics


# =========================
# CONFIG GCP (constantes)
# =========================

GCP_PROJECT = "dulcet-thinker-485219-q1"
GCP_ZONE = "us-central1-a"

CPU_MACHINE_TYPE = "e2-small"
CPU_IMAGE = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
CPU_DISK_SIZE_GB = 50
CPU_NETWORK = "global/networks/default"


DEPLOYMENT_NAME = "blockchain"
CPU_TAGS = ["worker-cpu"]

STARTUP_SCRIPT_PATH = "/scripts/worker_cpu_startup.sh"
MAX_CPU_WORKERS = 10
RABBIT_EXTERNAL_HOST = os.getenv("RABBIT_EXTERNAL_HOST")
COORD_EXTERNAL_HOST = os.getenv("COORDINADOR_EXTERNAL_HOST")
POOL_EXTERNAL_HOST = os.getenv("POOL_MANAGER_EXTERNAL_HOST")


# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cpu-scaler")

last_scale_time = 0
last_seen_cache = {}
# ---- hysteresis ----
cpu_below_expected_counter = 0
CPU_CONFIRM_CYCLES = 3  # cantidad de ciclos consecutivos


# ---------- Redis ----------
def redis_connect():
    r = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
    )
    r.ping()
    logger.info("Conectado a Redis")
    return r


def get_alive_workers(redis_client):
    alive = []
    for key in redis_client.scan_iter("worker:*"):
        data = redis_client.hgetall(key)
        if not data:
            continue

        # Normalizar tipos
        if "capacity" in data:
            data["capacity"] = int(data["capacity"])
        if "last_seen" in data:
            data["last_seen"] = float(data["last_seen"])

        alive.append(data)

    return alive


# ---------- GCP ----------
def load_startup_script():
    with open(STARTUP_SCRIPT_PATH) as f:
        return f.read()


def create_cpu_worker():
    client = compute_v1.InstancesClient()

    name = f"temp-{DEPLOYMENT_NAME}-worker-cpu-{int(time.time())}"

    instance = compute_v1.Instance(
        name=name,
        machine_type=f"zones/{GCP_ZONE}/machineTypes/{CPU_MACHINE_TYPE}",
        disks=[
            compute_v1.AttachedDisk(
                boot=True,
                auto_delete=True,
                initialize_params=compute_v1.AttachedDiskInitializeParams(
                    source_image=CPU_IMAGE,
                    disk_size_gb=CPU_DISK_SIZE_GB,
                ),
            )
        ],
        network_interfaces=[compute_v1.NetworkInterface(network=CPU_NETWORK)],
        metadata=compute_v1.Metadata(
            items=[
                compute_v1.Items(
                    key="startup-script",
                    value=load_startup_script(),
                ),
                compute_v1.Items(
                    key="COORDINADOR_HOST",
                    value=COORD_EXTERNAL_HOST,
                ),
                compute_v1.Items(
                    key="RABBIT_HOST",
                    value=RABBIT_EXTERNAL_HOST,
                ),
                compute_v1.Items(
                    key="POOL_MANAGER_HOST",
                    value=POOL_EXTERNAL_HOST,
                )
            ]
        ),
        tags=compute_v1.Tags(items=CPU_TAGS),
        labels={
            "role": "worker-cpu",
            "managed-by": "cpu-scaler",
        },
    )

    client.insert(
        project=GCP_PROJECT,
        zone=GCP_ZONE,
        instance_resource=instance,
    )

    logger.warning("Worker CPU dinámico creado: %s", name)


def list_dynamic_cpu_instances():
    client = compute_v1.InstancesClient()
    instances = []

    for inst in client.list(project=GCP_PROJECT, zone=GCP_ZONE):
        labels = inst.labels or {}
        if labels.get("managed-by") == "cpu-scaler":
            instances.append(inst.name)

    return instances


def delete_cpu_worker(instance_name):
    client = compute_v1.InstancesClient()

    client.delete(
        project=GCP_PROJECT,
        zone=GCP_ZONE,
        instance=instance_name,
    )

    logger.warning("Worker CPU dinámico eliminado: %s", instance_name)


def reconcile(redis_client):
    global last_scale_time
    global cpu_below_expected_counter

    now = time.time()

    # ---- métricas base ----
    metrics.reconciliations_total.inc()
    metrics.worker_info.clear()
    metrics.worker_heartbeat_age_seconds.clear()
    metrics.worker_heartbeat_interval_seconds.clear()

    alive = get_alive_workers(redis_client)
    logger.info("Workers vivos detectados: %d", len(alive))
    cpu_alive = 0
    gpu_alive = 0
    near_expiry_cpu = 0
    near_expiry_gpu = 0

    for w in alive:
        wid = w.get("id", "unknown")
        wtype = w.get("type", "unknown")
        last_seen = w.get("last_seen")

        if last_seen is None:
            continue

        # 1️⃣ Edad del heartbeat
        age = now - last_seen
        metrics.worker_heartbeat_age_seconds.labels(id=wid, type=wtype).set(age)

        # 2️⃣ Delta entre heartbeats
        prev = last_seen_cache.get(wid)
        if prev is not None:
            delta = last_seen - prev
            metrics.worker_heartbeat_interval_seconds.labels(id=wid, type=wtype).set(
                delta
            )
        last_seen_cache[wid] = last_seen

        # 3️⃣ Cerca de expirar
        if age > settings.HEARTBEAT_TTL * 0.7:
            if wtype == "cpu":
                near_expiry_cpu += 1
            elif wtype == "gpu":
                near_expiry_gpu += 1

        # Conteo por tipo
        if wtype == "cpu":
            cpu_alive += 1
        elif wtype == "gpu":
            gpu_alive += 1

        # Info por worker
        metrics.worker_info.labels(
            id=wid,
            type=wtype,
            ip=w.get("ip", "unknown"),
            capacity=str(w.get("capacity", "")),
        ).set(1)

    # ---- métricas agregadas ----
    metrics.total_workers_cpu.set(cpu_alive)
    metrics.total_workers_gpu.set(gpu_alive)
    metrics.total_workers.set(len(alive))
    metrics.workers_near_expiry.labels(type="cpu").set(near_expiry_cpu)
    metrics.workers_near_expiry.labels(type="gpu").set(near_expiry_gpu)

    # ---- cooldown global ----
    if now - last_scale_time < settings.SCALE_COOLDOWN:
        logger.info("Cooldown activo, no se evalúa escalado aún")
        return

    # ---- cálculo de demanda ----
    missing_gpus = max(settings.EXPECTED_GPUS - gpu_alive, 0)
    metrics.gpus_missing.set(missing_gpus)

    if missing_gpus > 0:
        target_cpu = settings.BASE_CPU_REPLICAS + (missing_gpus * settings.CPUS_PER_GPU)
        target_cpu = min(target_cpu, MAX_CPU_WORKERS)
    else:
        target_cpu = settings.BASE_CPU_REPLICAS

    metrics.target_cpu_workers.set(target_cpu)

    # ---- estado actual real ----
    dynamic_instances = list_dynamic_cpu_instances()
    metrics.dynamic_cpu_workers.set(len(dynamic_instances))

    effective_cpu = settings.BASE_CPU_REPLICAS + len(dynamic_instances)

    logger.info(
        "Estado actual: cpu_alive=%d gpu_alive=%d effective_cpu=%d target_cpu=%d dinámicos=%d",
        cpu_alive,
        gpu_alive,
        effective_cpu,
        target_cpu,
        len(dynamic_instances),
    )

    # =========================================================
    # HYSTERESIS: SOLO PARA SCALE-UP
    # =========================================================
    scale_up_allowed = True

    if effective_cpu < target_cpu:
        cpu_below_expected_counter += 1
        logger.info(
            "CPU efectiva baja (%d < %d). Confirmación %d/%d",
            effective_cpu,
            target_cpu,
            cpu_below_expected_counter,
            CPU_CONFIRM_CYCLES,
        )

        if cpu_below_expected_counter < CPU_CONFIRM_CYCLES:
            logger.info(
                "Faltante de CPU no confirmado (%d/%d). No se escala aún.",
                cpu_below_expected_counter,
                CPU_CONFIRM_CYCLES,
            )
            scale_up_allowed = False
    else:
        if cpu_below_expected_counter != 0:
            logger.info("CPU efectiva recuperada, reseteando contador de confirmación")
        cpu_below_expected_counter = 0

    # =========================================================
    # SCALE-UP (con hysteresis)
    # =========================================================
    if effective_cpu < target_cpu and scale_up_allowed:
        to_create = target_cpu - effective_cpu

        logger.warning(
            "Escalando CPU: creando %d workers (actual=%d, target=%d)",
            to_create,
            effective_cpu,
            target_cpu,
        )

        metrics.scale_up_total.inc()
        for _ in range(to_create):
            create_cpu_worker()
            metrics.vm_created_total.inc()

    # =========================================================
    # SCALE-DOWN PROPORCIONAL (SIN hysteresis)
    # =========================================================
    elif effective_cpu > target_cpu and dynamic_instances:
        to_delete = min(
            effective_cpu - target_cpu,
            len(dynamic_instances),
        )

        logger.warning(
            "Reduciendo CPU: eliminando %d workers dinámicos (actual=%d, target=%d)",
            to_delete,
            effective_cpu,
            target_cpu,
        )

        for name in dynamic_instances[:to_delete]:
            delete_cpu_worker(name)
            metrics.vm_deleted_total.inc()

    else:
        logger.info("No se requieren acciones de escalado en este ciclo")

    last_scale_time = now


# ---------- Main loop ----------
def main():
    redis_client = redis_connect()
    metrics.start_metrics_server(8000)
    logger.info("Conectado al coordinador en %s", COORD_EXTERNAL_HOST)
    logger.info("Conectado a RabbitMQ en %s", RABBIT_EXTERNAL_HOST)
    logger.info("Conectado al Pool Manager en %s", POOL_EXTERNAL_HOST)

    while True:
        metrics.update_uptime()
        try:
            redis_client.ping()
        except Exception:
            logger.warning("Redis connection perdida, reconectando...")
            redis_client = redis_connect()
        try:
            reconcile(redis_client)
        except Exception as e:
            metrics.errors_total.inc()
            logger.error("Error en cpu-scaler: %s", e)

        time.sleep(settings.SCALE_INTERVAL)


if __name__ == "__main__":
    main()

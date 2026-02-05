import time
import json
import logging
import redis

import config.settings as settings
from google.cloud import compute_v1

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

# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cpu-scaler")

last_scale_time = 0


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
    workers = []
    for key in redis_client.scan_iter("worker:*"):
        raw = redis_client.get(key)
        if not raw:
            continue
        workers.append(json.loads(raw))
    return workers


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


# ---------- Lógica de escalado ----------
def reconcile(redis_client):
    global last_scale_time

    now = time.time()
    if now - last_scale_time < settings.SCALE_COOLDOWN:
        return

    alive = get_alive_workers(redis_client)
    gpu_alive = [w for w in alive if w.get("type") == "gpu"]

    missing_gpus = max(settings.EXPECTED_GPUS - len(gpu_alive), 0)

    if missing_gpus > 0:
        target_cpu = settings.BASE_CPU_REPLICAS + (missing_gpus * settings.CPUS_PER_GPU)
        target_cpu = min(target_cpu, MAX_CPU_WORKERS)

        dynamic_instances = list_dynamic_cpu_instances()
        effective_cpu = settings.BASE_CPU_REPLICAS + len(dynamic_instances)

        to_create = target_cpu - effective_cpu

        if to_create > 0:
            logger.warning(
                "Faltan GPUs (%d). Creando %d workers CPU dinámicos",
                missing_gpus,
                to_create,
            )
            for _ in range(to_create):
                create_cpu_worker()

    else:
        # GPU volvió → eliminar TODOS los CPUs dinámicos
        dynamic_instances = list_dynamic_cpu_instances()

        if dynamic_instances:
            logger.warning(
                "GPU disponible. Eliminando %d workers CPU dinámicos",
                len(dynamic_instances),
            )
            for name in dynamic_instances:
                delete_cpu_worker(name)

    last_scale_time = now


# ---------- Main loop ----------
def main():
    redis_client = redis_connect()

    while True:
        try:
            reconcile(redis_client)
        except Exception as e:
            logger.error("Error en cpu-scaler: %s", e)

        time.sleep(settings.SCALE_INTERVAL)


if __name__ == "__main__":
    main()

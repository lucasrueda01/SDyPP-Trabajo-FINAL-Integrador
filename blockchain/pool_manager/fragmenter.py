import config.settings as settings


def fragmentar(block, num_gpus, num_cpus):
    """Divide el espacio de búsqueda de nonce en fragmentos para GPU y CPU.
    - GPUs: la parte de GPU se divide en N fragmentos (uno por GPU).
    - CPUs: el espacio restante se divide en N fragmentos de CPU.
    Devuelve una tupla: (gpu_payloads, cpu_payloads)
    donde gpu_payloads es una lista (puede estar vacía)."""

    nonce_start = block.get("nonce_start", 0)
    nonce_end = block.get(
        "nonce_end",
        block.get("numMaxRandom", settings.MAX_RANDOM),
    )

    total_space = nonce_end - nonce_start + 1

    gpu_capacity_total = num_gpus * settings.GPU_CAPACITY
    cpu_capacity_total = num_cpus * settings.CPU_CAPACITY
    total_capacity = gpu_capacity_total + cpu_capacity_total

    if total_capacity == 0:
        return [], []

    gpu_space = 0
    if gpu_capacity_total > 0:
        gpu_space = int(total_space * (gpu_capacity_total / total_capacity))

    cursor = nonce_start

    # ---- GPUs: split into N chunks ----
    gpu_payloads = []
    if num_gpus > 0 and gpu_space > 0:
        gpu_chunk = gpu_space // num_gpus
        gpu_cursor = cursor

        for i in range(num_gpus):
            start = gpu_cursor
            if i == num_gpus - 1:
                end = cursor + gpu_space - 1
            else:
                end = gpu_cursor + gpu_chunk - 1

            payload = {**block, "nonce_start": start, "nonce_end": end}
            gpu_payloads.append(payload)
            gpu_cursor = end + 1

        cursor = cursor + gpu_space

    # ---- CPUs: split remaining space ----
    cpu_payloads = []
    if num_cpus > 0 and cursor <= nonce_end:
        cpu_start = cursor
        cpu_end = nonce_end
        cpu_total_space = cpu_end - cpu_start + 1

        cpu_chunk = cpu_total_space // num_cpus
        cpu_cursor = cpu_start

        for i in range(num_cpus):
            start = cpu_cursor
            if i == num_cpus - 1:
                end = cpu_end
            else:
                end = cpu_cursor + cpu_chunk - 1

            payload = {**block, "nonce_start": start, "nonce_end": end}
            cpu_payloads.append(payload)
            cpu_cursor = end + 1

    return gpu_payloads, cpu_payloads

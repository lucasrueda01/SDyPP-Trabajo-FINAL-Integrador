import config.settings as settings


def fragmentar(block, num_gpus, num_cpus):

    nonce_start = block.get("nonce_start", 0)
    nonce_end = block.get(
        "nonce_end",
        block.get("numMaxRandom", settings.MAX_RANDOM),
    )

    total_space = nonce_end - nonce_start + 1

    if total_space <= 0:
        return [], []

    fragment_percent = block.get("fragment_percent", settings.FRAGMENT_PERCENT)

    fragment_size = int(total_space * fragment_percent)

    if fragment_size <= 0:
        fragment_size = 1

    gpu_payloads = []
    cpu_payloads = []

    cursor = nonce_start
    worker_index = 0
    total_workers = num_gpus + num_cpus

    if total_workers == 0:
        return [], []

    while cursor <= nonce_end:
        end = min(cursor + fragment_size - 1, nonce_end)

        payload = {**block, "nonce_start": cursor, "nonce_end": end}

        if worker_index < num_gpus:
            gpu_payloads.append(payload)
        else:
            cpu_payloads.append(payload)

        worker_index = (worker_index + 1) % total_workers
        cursor = end + 1

    return gpu_payloads, cpu_payloads

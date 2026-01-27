import subprocess
import json
import time
import sys
import logging
from pathlib import Path
import uuid

import config.settings as settings

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent

if sys.platform.startswith("win"):
    CUDA_BIN = BASE_DIR / "minero_cuda.exe"
else:
    CUDA_BIN = BASE_DIR / "minero_cuda"

if not CUDA_BIN.exists():
    raise RuntimeError(f"No se encontró el binario CUDA: {CUDA_BIN}")

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = logging.DEBUG if getattr(settings, "DEBUG", False) else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("minero_gpu")

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# -----------------------
# Minero
# -----------------------
def ejecutar_minero(from_val, to_val, prefix, hash_val):
    output_file = BASE_DIR / f"gpu_output_{uuid.uuid4().hex}.txt"
    start_time = time.time()

    cmd = [
        str(CUDA_BIN),
        str(from_val),
        str(to_val),
        prefix,
        hash_val,
        str(output_file),
    ]

    logger.info("Ejecutando CUDA: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        logger.error("Error ejecutando CUDA: %s", proc.stderr.strip())
        return json.dumps({"numero": 0, "hash_md5_result": "", "intentos": 0})

    contenido = output_file.read_text().strip() if output_file.exists() else ""

    try:
        if contenido:
            # formato: "<numero> <hash> <intentos>"
            numero_str, hash_str, intentos_str = contenido.split()
            resultado = {
                "numero": int(numero_str),
                "hash_md5_result": hash_str if int(numero_str) > 0 else "",
                "intentos": int(intentos_str),
            }
        else:
            resultado = {"numero": 0, "hash_md5_result": "", "intentos": 0}

    except Exception as e:
        logger.error("Error parseando salida CUDA: %s", e)
        resultado = {"numero": 0, "hash_md5_result": "", "intentos": 0}

    try:
        output_file.unlink(missing_ok=True)
    except Exception:
        pass

    logger.info(
        "GPU terminó rango %s-%s en %.2fs (intentos=%d)",
        from_val,
        to_val,
        time.time() - start_time,
        resultado["intentos"],
    )

    return json.dumps(resultado)

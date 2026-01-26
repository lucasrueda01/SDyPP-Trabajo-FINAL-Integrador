import os
import subprocess
import json
import time
import shutil
import sys
import logging
from pathlib import Path
import uuid

import config.settings as settings

BASE_DIR = Path(__file__).resolve().parent
CUDA_SOURCE = BASE_DIR / "minero_cuda.cu"
CUDA_OUTPUT = BASE_DIR / "minero_cuda"

# Tamaño de chunk que procesa la GPU por ejecución
GPU_CHUNK_SIZE = getattr(settings, "GPU_CHUNK_SIZE", 512 * 150)
# Timeout en segundos para cada ejecución del binario CUDA
CUDA_CHUNK_TIMEOUT = getattr(settings, "CUDA_CHUNK_TIMEOUT", 120)

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

# Silenciar ruido de librerías si las hubiera
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


# -----------------------
# Helpers
# -----------------------
def _nvcc_exists() -> bool:
    """Comprueba que `nvcc` esté en PATH."""
    return shutil.which("nvcc") is not None


def compile_cuda() -> bool:
    """
    Compila el fuente CUDA si no está compilado ya.
    Devuelve True si la binaria está disponible y ejecutable.
    """
    try:
        # Si ya existe la salida compilada y es ejecutable, devolvemos True
        if CUDA_OUTPUT.exists() and os.access(str(CUDA_OUTPUT), os.X_OK):
            logger.debug("Binario CUDA ya compilado y ejecutable: %s", CUDA_OUTPUT)
            return True

        if not _nvcc_exists():
            logger.error(
                "nvcc no encontrado en PATH. Instalá CUDA toolkit o pon nvcc en PATH."
            )
            return False

        logger.info("Compilando CUDA: %s -> %s", CUDA_SOURCE, CUDA_OUTPUT)
        compile_command = [
            "nvcc",
            str(CUDA_SOURCE),
            "-o",
            str(CUDA_OUTPUT),
            "-allow-unsupported-compiler",
        ]

        # Ejecutamos la compilación
        result = subprocess.run(compile_command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "Error compilando CUDA (returncode=%s). STDERR:\n%s",
                result.returncode,
                result.stderr.strip(),
            )
            return False

        # Aseguramos permisos de ejecución
        try:
            os.chmod(str(CUDA_OUTPUT), 0o755)
        except Exception:
            logger.debug("No se pudo cambiar permisos del binario (no crítico).")

        logger.info("CUDA compilado correctamente")
        return True

    except Exception:
        logger.exception("Excepción durante compilación CUDA")
        return False


def ejecutar_minero(from_val, to_val, prefix, hash_val):
    if not compile_cuda():
        return json.dumps({"numero": 0, "hash_md5_result": "", "intentos": 0})

    output_file = BASE_DIR / f"gpu_output_{uuid.uuid4().hex}.txt"
    start_time = time.time()

    cmd = [
        str(CUDA_OUTPUT),
        str(from_val),
        str(to_val),
        prefix,
        hash_val,
        str(output_file),
    ]

    logger.info("Comando CUDA a ejecutar: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        logger.error("Error ejecutando CUDA: %s", proc.stderr)
        return json.dumps({"numero": 0, "hash_md5_result": "", "intentos": 0})

    # Leer salida CUDA
    if output_file.exists():
        contenido = output_file.read_text().strip()
    else:
        contenido = ""

    try:
        if contenido:
            # Nuevo formato: "<numero> <hash> <intentos>"
            parts = contenido.split()
            if len(parts) != 3:
                raise ValueError(f"Formato inválido: {contenido}")

            numero_str, hash_str, intentos_str = parts

            resultado = {
                "numero": int(numero_str),
                "hash_md5_result": hash_str if int(numero_str) > 0 else "",
                "intentos": int(intentos_str),
            }
        else:
            resultado = {
                "numero": 0,
                "hash_md5_result": "",
                "intentos": 0,
            }

    except Exception as e:
        logger.error("Error parseando salida CUDA: %s", e)
        resultado = {
            "numero": 0,
            "hash_md5_result": "",
            "intentos": 0,
        }

    # Limpieza del archivo temporal
    try:
        if output_file.exists():
            os.remove(output_file)
    except Exception:
        pass

    logger.info(
        "GPU terminó rango %s-%s en %.2fs (intentos reales=%d)",
        from_val,
        to_val,
        time.time() - start_time,
        resultado["intentos"],
    )

    return json.dumps(resultado)


if __name__ == "__main__":
    print("Ejecutando minero GPU de prueba...")
    out = ejecutar_minero(1, 99_999_999, "0000000", "HOLAMUNDO")
    print(out)

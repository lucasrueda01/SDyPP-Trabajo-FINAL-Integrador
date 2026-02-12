%%writefile minero_gpu.py
import subprocess
import time
import sys
from pathlib import Path
import uuid
import json

BASE_DIR = Path(__file__).resolve().parent

if sys.platform.startswith("win"):
    CUDA_BIN = BASE_DIR / "minero_cuda.exe"
else:
    CUDA_BIN = BASE_DIR / "minero_cuda"

if not CUDA_BIN.exists():
    raise RuntimeError(f"No se encontro el binario CUDA: {CUDA_BIN}")

def ejecutar_minero(from_val, to_val, prefix, input_val):
    """
    Ejecuta el binario CUDA y devuelve un dict con:
      { "numero": int, "hash_md5_result": str, "intentos": int }
    Igual formato que antes (sin logging).
    """
    output_file = BASE_DIR / f"gpu_output_{uuid.uuid4().hex}.txt"
    start_time = time.time()

    cmd = [
        str(CUDA_BIN),
        str(from_val),
        str(to_val),
        prefix,
        input_val,
        str(output_file),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Si falla el binario devolvemos la misma estructura vacia
    if proc.returncode != 0:
        return {"numero": 0, "hash_md5_result": "", "intentos": 0}

    contenido = output_file.read_text().strip() if output_file.exists() else ""

    try:
        if contenido:
            numero_str, hash_str, intentos_str = contenido.split()
            resultado = {
                "numero": int(numero_str),
                "hash_md5_result": hash_str if int(numero_str) > 0 else "",
                "intentos": int(intentos_str),
            }
        else:
            resultado = {"numero": 0, "hash_md5_result": "", "intentos": 0}
    except Exception:
        resultado = {"numero": 0, "hash_md5_result": "", "intentos": 0}

    # intento de borrar el archivo temporal
    try:
        output_file.unlink(missing_ok=True)
    except Exception:
        pass

    return resultado
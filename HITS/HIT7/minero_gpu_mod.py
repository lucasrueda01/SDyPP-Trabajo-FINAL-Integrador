%%writefile minero_gpu_mod.py
import subprocess
import sys
from pathlib import Path
import uuid

BASE_DIR = Path(".").resolve()

if sys.platform.startswith("win"):
    CUDA_BIN = BASE_DIR / "minero_cuda_modificado.exe"
else:
    CUDA_BIN = BASE_DIR / "minero_cuda_modificado"

if not CUDA_BIN.exists():
    raise RuntimeError(f"No se encontro el binario CUDA: {CUDA_BIN}")

def ejecutar_minero(prefix, input_val, from_=0, to_=200000):
    """
    Ejecuta el binario CUDA pasando el prefijo, la cadena y el rango [from_, to_].
    Retorna dict: {"numero": int, "hash_md5_result": str, "intentos": int}
    """
    output_file = BASE_DIR / f"gpu_output_{uuid.uuid4().hex}.txt"

    cmd = [
        str(CUDA_BIN),
        prefix,
        input_val,
        str(output_file),
        str(int(from_)),
        str(int(to_)),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return {"numero": 0, "hash_md5_result": "", "intentos": 0}

    contenido = ""
    try:
        contenido = output_file.read_text().strip()
    except Exception:
        return {"numero": 0, "hash_md5_result": "", "intentos": 0}

    try:
        numero, hash_md5, intentos = contenido.split()
        resultado = {
            "numero": int(numero),
            "hash_md5_result": hash_md5 if int(numero) > 0 else "",
            "intentos": int(intentos),
        }
    except Exception:
        resultado = {"numero": 0, "hash_md5_result": "", "intentos": 0}

    try:
        output_file.unlink()
    except Exception:
        pass

    return resultado


if __name__ == "__main__":
    res = ejecutar_minero("00", "hola mundo", 0, 200000)
    print(res)
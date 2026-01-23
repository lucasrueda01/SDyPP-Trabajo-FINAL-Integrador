import os
import subprocess
import json
import time
from pathlib import Path
import uuid

BASE_DIR = Path(__file__).resolve().parent
CUDA_SOURCE = BASE_DIR / "minero_cuda.cu"
CUDA_OUTPUT = BASE_DIR / "minero_cuda"
# Tamaño de chunk que procesa la GPU por ejecución
GPU_CHUNK_SIZE = 512 * 150


def compile_cuda():
    if CUDA_OUTPUT.exists():
        return True

    compile_command = [
        "nvcc",
        str(CUDA_SOURCE),
        "-o",
        str(CUDA_OUTPUT),
        "-allow-unsupported-compiler",
    ]

    result = subprocess.run(compile_command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error compilando CUDA:")
        print(result.stderr)
        return False

    print("CUDA compilado correctamente")
    return True


def ejecutar_minero(from_val, to_val, prefix, hash_val):
    """
    Devuelve SIEMPRE un JSON:
    {
        "numero": int,
        "hash_md5_result": str,
        "intentos": int
    }
    """

    if not compile_cuda():
        return json.dumps({
            "numero": 0,
            "hash_md5_result": "",
            "intentos": 0
        })

    output_file = BASE_DIR / f"gpu_output_{uuid.uuid4().hex}.txt"

    desde = from_val
    encontrado = False
    intentos_totales = 0

    start_time = time.time()

    while desde < to_val and not encontrado:
        hasta = min(desde + GPU_CHUNK_SIZE, to_val)

        # Llamada al binario CUDA (CONTRATO NUEVO)
        cmd = [
            str(CUDA_OUTPUT),
            str(desde),
            str(hasta),
            prefix,
            hash_val,
            str(output_file)
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if proc.returncode != 0:
            print(" Error ejecutando CUDA:")
            print(proc.stderr)
            break

        # Intentos estimados del chunk
        intentos_totales += (hasta - desde)

        # Leer salida CUDA
        if output_file.exists():
            contenido = output_file.read_text().strip()
        else:
            contenido = ""

        if contenido:
            # Esperamos formato: "<nonce> <hash>"
            try:
                numero_str, hash_str = contenido.split()
                encontrado = True
                resultado = {
                    "numero": int(numero_str),
                    "hash_md5_result": hash_str,
                    "intentos": intentos_totales
                }
                break
            except ValueError:
                print("Formato inválido de salida CUDA:", contenido)
                break

        # Avanzar rango
        desde = hasta

    execution_time = time.time() - start_time

    # Limpieza del archivo temporal
    try:
        if output_file.exists():
            os.remove(output_file)
    except:
        pass

    if not encontrado:
        resultado = {
            "numero": 0,
            "hash_md5_result": "",
            "intentos": intentos_totales
        }

    return json.dumps(resultado)


# Ejemplo de uso
if __name__ == "__main__":
    ejecutar_minero(1, 10000000, "00000", "PAPA")

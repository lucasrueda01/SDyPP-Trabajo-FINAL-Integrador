%%writefile minero_cpu_hit_7.py
import hashlib
import argparse

# valores por defecto
DEFAULT_RANGE_START = 0
DEFAULT_RANGE_END = 200000

def ejecutar_minero(prefix, input_val, range_start=DEFAULT_RANGE_START, range_end=DEFAULT_RANGE_END):
    
    intentos = 0

    for nonce in range(range_start, range_end + 1):
        mensaje = f"{nonce}{input_val}".encode("utf-8")
        hash_md5 = hashlib.md5(mensaje).hexdigest()
        intentos += 1

        if hash_md5.startswith(prefix):
            return {
                "numero": nonce,
                "hash_md5_result": hash_md5,
                "intentos": intentos,
            }

    # Si no se encuentra nada
    return {
        "numero": 0,
        "hash_md5_result": "",
        "intentos": intentos,
    }

def _crear_parser():
    p = argparse.ArgumentParser(description="Minero CPU por fuerza bruta (MD5).")
    p.add_argument("--prefix", "-p", default="00", help='Prefijo que debe cumplir el hash (ej: "00")')
    p.add_argument("--input", "-i", dest="input_val", default="hola mundo", help='Texto a concatenar con el nonce')
    p.add_argument("--start", "-s", type=int, default=DEFAULT_RANGE_START, help="Nonce inicial (inclusive)")
    p.add_argument("--end", "-e", type=int, default=DEFAULT_RANGE_END, help="Nonce final (inclusive)")
    return p

if __name__ == "__main__":
    parser = _crear_parser()
    args = parser.parse_args()

    resultado = ejecutar_minero_cpu(args.prefix, args.input_val, args.start, args.end)
    print(resultado)
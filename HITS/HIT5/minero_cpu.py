%%writefile minero_cpu.py
import hashlib

def ejecutar_minero_cpu(from_val, to_val, prefix, input_val):
    """
    Version CPU.
    Formato:
      { "numero": int, "hash_md5_result": str, "intentos": int }
    """

    intentos = 0

    for nonce in range(from_val, to_val + 1):
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
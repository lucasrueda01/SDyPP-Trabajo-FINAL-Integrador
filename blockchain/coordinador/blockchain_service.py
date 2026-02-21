import hashlib
import time


def calculateHash(data):
    h = hashlib.md5()
    h.update(data.encode("utf-8"))
    return h.hexdigest()


def validarTransaction(transaction):
    required = ["origen", "destino", "monto"]

    # Verificar que existan las keys
    if not all(k in transaction for k in required):
        return False

    origen = transaction["origen"]
    destino = transaction["destino"]
    monto = transaction["monto"]

    # Validar strings no vacíos
    if not isinstance(origen, str) or not origen.strip():
        return False

    if not isinstance(destino, str) or not destino.strip():
        return False

    # Validar monto
    if not isinstance(monto, (int, float)):
        return False

    if monto <= 0:
        return False

    return True


def construirNuevoBloque(block, prev, result_hash, nonce):
    return {
        "blockId": block["blockId"],
        "hash": result_hash,
        "hashPrevio": prev["hash"] if prev else None,
        "nonce": nonce,
        "prefijo": block["prefijo"],
        "transactions": block["transactions"],
        "timestamp": time.time(),
        "baseStringChain": block["baseStringChain"],
        "blockchainContent": calculateHash(block["baseStringChain"] + result_hash),
    }

import hashlib
import time


def calculateHash(data):
    h = hashlib.md5()
    h.update(data.encode("utf-8"))
    return h.hexdigest()


def validarTransaction(transaction):
    required = ["origen", "destino", "monto"]
    return all(k in transaction and transaction[k] for k in required)


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

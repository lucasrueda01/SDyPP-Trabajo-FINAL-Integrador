import json
import logging
from google.cloud import storage
from google.api_core.exceptions import NotFound

logger = logging.getLogger("coordinator")


def bucketConnect(bucketName):
    client = storage.Client()
    logger.info("Conectado a Google Cloud Storage")
    return client.bucket(bucketName)


def subirBlock(bucket, block):
    logger.debug(f"Subiendo bloque {block['blockId']} a storage")
    blob = bucket.blob(f"block_{block['blockId']}.json")
    blob.upload_from_string(json.dumps(block), content_type="application/json")


def descargarBlock(bucket, blockId):
    blob = bucket.blob(f"block_{blockId}.json")
    try:
        return json.loads(blob.download_as_text())
    except NotFound:
        return None


def borrarBlock(bucket, blockId):
    blob = bucket.blob(f"block_{blockId}.json")
    try:
        blob.delete()
    except NotFound:
        pass

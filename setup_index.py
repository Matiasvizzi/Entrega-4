"""Inicialización: crea el índice Serverless de Pinecone si todavía no existe.

Uso:
    python setup_index.py
"""
import time

from pinecone import Pinecone, ServerlessSpec

from config import (
    EMBEDDING_DIM,
    INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    get_pinecone_api_key,
)


def get_client() -> Pinecone:
    return Pinecone(api_key=get_pinecone_api_key())


def asegurar_indice(pc: Pinecone | None = None):
    """Crea el índice si no existe y devuelve el objeto Index listo para usar."""
    pc = pc or get_client()
    existentes = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in existentes:
        print(f"🆕 Creando índice serverless '{INDEX_NAME}' (dim={EMBEDDING_DIM}, cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)
        print("✅ Índice listo")
    else:
        dim = pc.describe_index(INDEX_NAME).dimension
        if dim != EMBEDDING_DIM:
            raise SystemExit(
                f"❌ El índice '{INDEX_NAME}' tiene dimensión {dim} pero el modelo de "
                f"embeddings produce {EMBEDDING_DIM}. Usá otro INDEX_NAME o borrá el índice."
            )
        print(f"♻️ El índice '{INDEX_NAME}' ya existe (dim={dim}) — no se vuelve a crear")

    return pc.Index(INDEX_NAME)


if __name__ == "__main__":
    indice = asegurar_indice()
    print(indice.describe_index_stats())

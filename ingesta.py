"""Pipeline de ingesta: /data -> chunking -> embeddings -> Pinecone Serverless.

El texto original se guarda dentro de la metadata del vector (campo "text"), junto
con fuente, página y categoría, así una sola consulta devuelve vector + texto + fuente
sin necesidad de una base de datos aparte.

Uso:
    python ingesta.py            # crea el índice si falta e indexa (idempotente)
    python ingesta.py --reset    # borra el namespace y reindexa desde cero
"""
import argparse
import time
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    NAMESPACE,
)
from setup_index import asegurar_indice


def get_embeddings() -> HuggingFaceEmbeddings:
    """Mismo modelo para indexar y para consultar."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def cargar_chunks():
    """Lee .txt/.md de /data, los fragmenta y agrega metadata avanzada.

    También lo usa el retriever BM25 (que vive en memoria) para trabajar sobre
    exactamente los mismos fragmentos que están en Pinecone.
    """
    documentos = []
    for patron in ("*.txt", "*.md"):
        loader = DirectoryLoader(
            str(DATA_DIR), glob=patron, loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documentos.extend(loader.load())
    if not documentos:
        raise FileNotFoundError(f"No hay archivos .txt/.md en {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documentos)

    contador_por_archivo: dict[str, int] = {}
    for chunk in chunks:
        nombre = Path(chunk.metadata["source"]).name
        n = contador_por_archivo.get(nombre, 0)
        contador_por_archivo[nombre] = n + 1
        chunk.metadata = {
            "source": nombre,
            "pagina": 1,  # los .txt/.md no tienen paginación
            "categoria": Path(nombre).stem.replace("_", " "),
            "chunk_id": n,
        }
    return chunks


def ingestar(reset: bool = False) -> PineconeVectorStore:
    indice = asegurar_indice()
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(
        index=indice, embedding=embeddings, namespace=NAMESPACE, text_key="text"
    )

    if reset:
        stats = indice.describe_index_stats()
        if NAMESPACE in stats["namespaces"]:
            print(f"🗑️ --reset: se borra el namespace '{NAMESPACE}'")
            indice.delete(delete_all=True, namespace=NAMESPACE)

    chunks = cargar_chunks()
    # IDs determinísticos (archivo::chunk): reejecutar la ingesta actualiza los
    # mismos vectores en vez de duplicarlos.
    ids = [f"{c.metadata['source']}::{c.metadata['chunk_id']}" for c in chunks]
    print(f"✂️ {len(chunks)} fragmentos -> subiendo a Pinecone (namespace '{NAMESPACE}')...")
    vectorstore.add_documents(chunks, ids=ids)

    # Pinecone es eventualmente consistente: esperar a que los vectores sean visibles
    for _ in range(30):
        total = indice.describe_index_stats()["namespaces"].get(NAMESPACE, {}).get("vector_count", 0)
        if total >= len(chunks):
            break
        time.sleep(2)
    print(f"📦 Vectores en el namespace '{NAMESPACE}': {total}")
    return vectorstore


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingesta de documentos en Pinecone")
    ap.add_argument("--reset", action="store_true", help="borrar el namespace y reindexar")
    ingestar(reset=ap.parse_args().reset)

"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) con EnsembleRetriever.

Uso:
    python rag_system.py "¿Qué es el MDM?"
    python rag_system.py --interactivo
"""
import argparse
import re
from typing import Dict, List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore

from config import ENSEMBLE_WEIGHTS, INDEX_NAME, NAMESPACE, TOP_K, get_pinecone_api_key
from ingesta import cargar_chunks, get_embeddings


def _tokenizar(texto: str) -> List[str]:
    # minúsculas y sin puntuación: que "2FA?" y "2fa" coincidan en BM25
    return re.findall(r"\w+", texto.lower())


class RAGSystem:
    """Encapsula un retriever (por defecto el EnsembleRetriever) y devuelve el top-k."""

    def __init__(self, retriever, k: int = TOP_K):
        self.retriever = retriever
        self.k = k

    # ---------------------------------------------------------- construcción
    @classmethod
    def crear(cls, k: int = TOP_K, weights=None, modo: str = "hibrido") -> "RAGSystem":
        """modo: 'hibrido' (BM25 + vectorial), 'bm25' o 'vectorial' (para comparar)."""
        chunks = cargar_chunks()

        bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_tokenizar)
        bm25.k = k

        get_pinecone_api_key()  # falla con mensaje claro si falta
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME, embedding=get_embeddings(),
            namespace=NAMESPACE, text_key="text",
        )
        vectorial = vectorstore.as_retriever(search_kwargs={"k": k})

        if modo == "bm25":
            retriever = bm25
        elif modo == "vectorial":
            retriever = vectorial
        else:
            retriever = EnsembleRetriever(
                retrievers=[bm25, vectorial], weights=weights or ENSEMBLE_WEIGHTS
            )
        sistema = cls(retriever, k=k)
        sistema.chunks = chunks
        return sistema

    # --------------------------------------------------------------- consulta
    def obtener_top_k(self, query: str) -> List[Dict]:
        docs = self.retriever.invoke(query)[: self.k]
        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("source", "desconocida"),
                "categoria": d.metadata.get("categoria", "desconocida"),
            }
            for d in docs
        ]


def _imprimir(resultados: List[Dict]) -> None:
    if not resultados:
        print("⚠️ No se recuperó ningún fragmento.\n")
        return
    print(f"\n📎 Top {len(resultados)} fragmentos:")
    for i, r in enumerate(resultados, 1):
        extracto = r["contenido"][:200].replace("\n", " ")
        print(f"  {i}. [{r['fuente']} | {r['categoria']}]\n     {extracto}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Recuperador híbrido BM25 + Pinecone")
    ap.add_argument("consulta", nargs="?", help="consulta única")
    ap.add_argument("--interactivo", action="store_true")
    args = ap.parse_args()

    rag = RAGSystem.crear()
    if args.consulta:
        _imprimir(rag.obtener_top_k(args.consulta))
    elif args.interactivo:
        print("💬 Consultá el sistema ('salir' para terminar)\n")
        while (q := input("🧑 Vos: ").strip()) and q.lower() not in ("salir", "exit", "quit"):
            _imprimir(rag.obtener_top_k(q))
    else:
        ap.print_help()

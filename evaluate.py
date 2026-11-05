"""Evaluación del recuperador: Recall@k y Precision@k sobre un Golden Set.

Uso:
    python evaluate.py              # evalúa el recuperador híbrido
    python evaluate.py --comparar   # compara BM25 vs vectorial vs híbrido
"""
import argparse
import json
from typing import Dict, List

from config import GOLDEN_SET_PATH, TOP_K
from rag_system import RAGSystem


def cargar_golden_set(path=GOLDEN_SET_PATH) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluar(rag: RAGSystem, golden_set: List[Dict], k: int = TOP_K) -> Dict:
    # chunks que tiene cada documento en el corpus: determina el máximo de Precision@k
    chunks_por_doc: Dict[str, int] = {}
    for c in rag.chunks:
        chunks_por_doc[c.metadata["source"]] = chunks_por_doc.get(c.metadata["source"], 0) + 1

    detalle = []
    for caso in golden_set:
        esperado = caso["documento_id_esperado"]
        top_k = rag.obtener_top_k(caso["pregunta"])[:k]
        fuentes = [r["fuente"] for r in top_k]

        # Recall@k: ¿el documento esperado aparece entre los k recuperados? (0 o 1)
        recall = 1.0 if esperado in fuentes else 0.0
        # Precision@k: fracción de los k recuperados que son útiles (vienen del doc esperado)
        precision = sum(f == esperado for f in fuentes) / k
        # Máximo alcanzable: si el doc tiene solo 1 chunk, nunca podrá superar 1/k
        maximo = min(k, chunks_por_doc.get(esperado, 0)) / k
        detalle.append({
            "pregunta": caso["pregunta"], "esperado": esperado, "recuperados": fuentes,
            "recall": recall, "precision": precision, "precision_max": maximo,
        })

    n = len(detalle)
    return {
        "detalle": detalle,
        "recall": sum(d["recall"] for d in detalle) / n,
        "precision": sum(d["precision"] for d in detalle) / n,
        "precision_max": sum(d["precision_max"] for d in detalle) / n,
    }


def imprimir_reporte(titulo: str, rep: Dict, k: int, detallado: bool = True) -> None:
    print("=" * 80)
    print(f"📋 {titulo}")
    print("=" * 80)
    if detallado:
        for d in rep["detalle"]:
            icono = "✅" if d["recall"] == 1.0 else "❌"
            print(f"{icono} {d['pregunta']}")
            print(f"   Esperado: {d['esperado']}")
            print(f"   Recuperados: {d['recuperados']}")
            print(f"   Recall@{k}: {d['recall']:.0%} | Precision@{k}: {d['precision']:.0%} "
                  f"(máx. posible: {d['precision_max']:.0%})\n")
    print(f"📊 Recall@{k} promedio:    {rep['recall']:.1%}")
    print(f"📊 Precision@{k} promedio: {rep['precision']:.1%}  (máx. alcanzable: {rep['precision_max']:.1%})\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluación Precision@k / Recall@k")
    ap.add_argument("--comparar", action="store_true", help="BM25 vs vectorial vs híbrido")
    args = ap.parse_args()

    golden = cargar_golden_set()
    print(f"Golden Set: {len(golden)} preguntas | k = {TOP_K}\n")

    if args.comparar:
        for modo in ("bm25", "vectorial", "hibrido"):
            rag = RAGSystem.crear(modo=modo)
            imprimir_reporte(f"Recuperador: {modo.upper()}", evaluar(rag, golden), TOP_K, detallado=False)
    else:
        rag = RAGSystem.crear()
        imprimir_reporte("Recuperador HÍBRIDO (BM25 + Pinecone)", evaluar(rag, golden), TOP_K)

# Sistema RAG escalable en la nube con Pinecone — Pre-entrega 4

Módulo de recuperación **híbrido** (búsqueda semántica en Pinecone Serverless + búsqueda léxica BM25) sobre las políticas internas de una empresa ficticia (TechCorp), con un script de evaluación de **Precision@5** y **Recall@5**.

## Arquitectura

```
data/*.txt ─► ingesta.py ─► chunking (600 tok, overlap 100) ─► embeddings locales (MiniLM, 384 dim)
                                                                      │
                                    Pinecone Serverless (namespace "politicas-internas")
                                    vector + metadata {text, source, pagina, categoria, chunk_id}
                                                                      │
consulta ─► RAGSystem ─► EnsembleRetriever ◄─ BM25Retriever (en memoria, mismos chunks)
                              │  (pesos 0.5 / 0.5, fusión RRF)  ◄─ Retriever vectorial (Pinecone)
                              ▼
                          top-5 fragmentos ─► evaluate.py (Recall@5, Precision@5)
```

| Archivo | Rol |
|---|---|
| `setup_index.py` | Verifica si el índice existe y lo crea en modo Serverless si falta. Valida que la dimensión coincida. |
| `ingesta.py` | Pipeline: carga `/data`, fragmenta, genera embeddings y sube a Pinecone con metadata avanzada. |
| `rag_system.py` | Clase `RAGSystem` que encapsula el `EnsembleRetriever` y expone `obtener_top_k(query)`. |
| `evaluate.py` | Calcula Recall@5 y Precision@5 sobre `golden_set.json`. |
| `golden_set.json` | 8 pares `{"pregunta", "documento_id_esperado"}`. |
| `config.py` | Parámetros compartidos (índice, namespace, chunking, dimensión). |
| `data/` | 7 documentos de ejemplo (`.txt`). |

## Decisiones de diseño (y errores comunes evitados)

- **Dimensiones:** `EMBEDDING_DIM = 384` se define una sola vez en `config.py` (modelo `all-MiniLM-L6-v2`, local y gratuito) y se usa para crear el índice. Si el índice ya existe con otra dimensión, el script falla con un mensaje claro en vez de subir vectores incompatibles. (1536 solo aplica a `text-embedding-3-small` de OpenAI.)
- **Namespace:** todo se indexa y consulta en `politicas-internas`, para separar tipos de datos.
- **Chunking:** 600 tokens / 100 de overlap, medidos con el tokenizer (`from_tiktoken_encoder`), dentro del rango sugerido de 500–800.
- **Texto en la metadata:** cada vector guarda su `text` original, más `source`, `pagina`, `categoria` y `chunk_id`, así no hace falta una base de datos aparte.
- **Ingesta idempotente:** los IDs son determinísticos (`archivo::n`), por lo que reejecutar la ingesta actualiza los vectores en vez de duplicarlos. `--reset` borra el namespace y reindexa.
- **BM25 sobre los mismos chunks:** se construye en memoria desde `/data` con los mismos fragmentos que están en Pinecone, con un tokenizador que ignora mayúsculas y puntuación (para que `MDM?` coincida con `mdm`).
- **Fusión híbrida:** `EnsembleRetriever` con pesos `[0.5, 0.5]` (editables en `config.py`). BM25 aporta en siglas y nombres exactos (2FA, MDM, VPN, GlobalConnect); el vectorial en paráfrasis.

## Cómo replicar el índice

Requiere Python 3.10+ y una cuenta gratuita en [Pinecone](https://app.pinecone.io).

```bash
# 1. Entorno virtual y dependencias
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate en Linux/macOS)
pip install -r requirements.txt

# 2. Variables de entorno
copy .env.example .env            # Windows  (cp en Linux/macOS)
# editar .env: PINECONE_API_KEY, INDEX_NAME, NAMESPACE

# 3. Crear el índice Serverless (si no existe)
python setup_index.py

# 4. Ingesta: chunking + embeddings + subida a Pinecone
python ingesta.py                 # idempotente; --reset para reindexar desde cero

# 5. Probar el recuperador
python rag_system.py "¿Qué significa MDM?"
python rag_system.py --interactivo

# 6. Evaluación
python evaluate.py                # recuperador híbrido
python evaluate.py --comparar     # BM25 vs vectorial vs híbrido
```

La primera ejecución descarga el modelo de embeddings (~90 MB). `ingesta.py` ejecuta `setup_index.py` automáticamente, por lo que el paso 3 es opcional.

## Evaluación

`evaluate.py` recorre el Golden Set y calcula, con k = 5:

- **Recall@5:** 1 si el documento esperado aparece entre los 5 recuperados, 0 si no. Es binario porque cada pregunta tiene un único documento esperado.
- **Precision@5:** fracción de los 5 recuperados que provienen del documento esperado.

> **Límite de Precision@5:** si el documento esperado tiene un solo chunk, la precisión máxima posible es 1/5 = 20 %, aunque el recuperador sea perfecto. Por eso el reporte muestra también el *máximo alcanzable*. Con este dataset pequeño (7 documentos, 1 chunk cada uno) ese techo es 20 %.

### Resultados

Salida real de `python evaluate.py --comparar` (índice en Pinecone Serverless, 7 vectores, 8 preguntas, k = 5):

| Recuperador | Recall@5 | Precision@5 | Máx. Precision@5 |
|---|---|---|---|
| BM25 | 100 % | 20 % | 20 % |
| Vectorial (Pinecone) | 100 % | 20 % | 20 % |
| Híbrido | 100 % | 20 % | 20 % |

Los tres recuperadores encuentran el documento correcto en las 8 preguntas, y la precisión alcanza el máximo posible del dataset. **Con 7 chunks en total y k = 5, la evaluación no distingue entre los métodos**: el top-5 abarca casi todo el corpus. Para ver la ventaja del recuperador híbrido hace falta un corpus más grande (decenas de documentos o varios chunks por documento) y preguntas más difíciles.
## Seguridad

La API key se lee de `.env`, que está en `.gitignore`. Solo se sube `.env.example`.

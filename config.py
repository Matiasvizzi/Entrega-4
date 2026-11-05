"""Configuración compartida por setup, ingesta, recuperador y evaluación."""
import os
from pathlib import Path

from dotenv import load_dotenv

# En redes corporativas con inspección TLS, usar el almacén de certificados del
# sistema evita errores CERTIFICATE_VERIFY_FAILED. Es opcional.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
GOLDEN_SET_PATH = BASE_DIR / "golden_set.json"

# --- Pinecone ---
INDEX_NAME = os.getenv("INDEX_NAME", "techcorp-rag-hibrido")
NAMESPACE = os.getenv("NAMESPACE", "politicas-internas")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Embeddings (locales) ---
# La dimensión del índice DEBE coincidir con la del modelo (evita el "mismatch de
# dimensiones"). all-MiniLM-L6-v2 -> 384. (1536 sería para OpenAI text-embedding-3-small.)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking (en tokens) ---
CHUNK_SIZE = 600      # punto medio del rango sugerido 500-800
CHUNK_OVERLAP = 100

# --- Recuperación ---
TOP_K = 5
ENSEMBLE_WEIGHTS = [0.5, 0.5]  # [BM25, vectorial]


def get_pinecone_api_key() -> str:
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise SystemExit(
            "❌ Falta PINECONE_API_KEY. Copiá .env.example a .env y completalo "
            "(gratis en https://app.pinecone.io)."
        )
    return key

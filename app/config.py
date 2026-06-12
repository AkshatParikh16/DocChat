import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM ────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_0")

    # ── Pinecone ────────────────────────────────────────────
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "policychat")

    # ── Embedding model ─────────────────────────────────────
    # BAAI/bge-small-en-v1.5: 384-dim, free, local, outperforms all-MiniLM-L6-v2
    # on retrieval benchmarks while keeping the same Pinecone index dimension.
    # NOTE: changing this requires re-ingesting all documents (embedding spaces differ).
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DIMENSION: int = 384

    # ── Chunking ────────────────────────────────────────────
    # CHUNK_TOKENS is used by Docling's HybridChunker (token-budget per chunk).
    # CHUNK_SIZE / CHUNK_OVERLAP are the char-based fallback for plain TXT files.
    CHUNK_TOKENS: int = 512
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # HuggingFace tokenizer name for HybridChunker — should match EMBEDDING_MODEL.
    CHUNKER_TOKENIZER: str = os.getenv("CHUNKER_TOKENIZER", "BAAI/bge-small-en-v1.5")

    # ── Retrieval ───────────────────────────────────────────
    TOP_K: int = 5
    BM25_TOP_K: int = 5
    CONFIDENCE_THRESHOLD: float = 0.35

    # ── Reranker ────────────────────────────────────────────
    # Cross-encoder reranking: free, local, significant precision boost.
    # Runs after hybrid scoring to re-order the candidate pool before LLM call.
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ── Paths ───────────────────────────────────────────────
    DATA_DIR: str = "data"
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"
    DOCUMENTS_JSON_PATH: str = "data/documents.json"

    # ── Logging ─────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    APP_ENV: str = os.getenv("APP_ENV", "development")


config = Config()

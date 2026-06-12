from sentence_transformers import SentenceTransformer
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)

# BGE models require a task-specific instruction prefix on queries (not documents).
# This significantly improves retrieval quality for these models.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_MODEL_PREFIXES = ("BAAI/bge-",)

model = None


def get_embedding_model() -> SentenceTransformer:
    """Load embedding model once (singleton)."""
    global model
    if model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return model


def _needs_query_prefix(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _BGE_MODEL_PREFIXES)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed document chunks for ingestion.
    Documents are embedded as-is (no instruction prefix).
    """
    embedding_model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    logger.info(f"Embedding complete: {len(chunks)} chunks")
    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embed a user query for similarity search.
    BGE models get the retrieval instruction prefix applied automatically.
    """
    embedding_model = get_embedding_model()

    text = query
    if _needs_query_prefix(config.EMBEDDING_MODEL):
        text = _BGE_QUERY_PREFIX + query

    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

from pinecone import Pinecone
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)

# Singleton Pinecone client
pc = None
index = None

def get_pinecone_index():
    """Get or create Pinecone index connection (singleton)."""
    global pc, index
    
    if index is None:
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone index: {config.PINECONE_INDEX_NAME}")
    
    return index


def upsert_chunks(chunks: list[dict]) -> bool:
    """
    Store embedded chunks in Pinecone.
    
    Args:
        chunks: Chunks with embeddings from embed_chunks()
    
    Returns:
        True if successful
    """
    try:
        pinecone_index = get_pinecone_index()
        
        vectors = []
        for chunk in chunks:
            meta = chunk["metadata"]
            vectors.append({
                "id": chunk["chunk_id"],
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "source": meta["source"],
                    "file_type": meta["file_type"],
                    "chunk_index": meta["chunk_index"],
                    "total_chunks": meta.get("total_chunks"),
                    "doc_id": chunk["doc_id"],
                    # structural metadata from Docling
                    "headings": " > ".join(meta.get("headings") or []),
                    "pages": meta.get("pages") or [],
                    "chunk_type": meta.get("chunk_type", "text"),
                }
            })
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            pinecone_index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
        
        logger.info(f"Total upserted: {len(vectors)} vectors")
        return True
        
    except Exception as e:
        logger.error(f"Pinecone upsert failed: {e}")
        raise


def search_pinecone(query_embedding: list[float],
                    top_k: int = None,
                    doc_id: str = None) -> list[dict]:
    """
    Search Pinecone for similar chunks.
    
    Args:
        query_embedding: Embedded query vector
        top_k: Number of results to return
        doc_id: Optional filter to search only specific document
    
    Returns:
        List of matching chunks with scores
    """
    try:
        pinecone_index = get_pinecone_index()
        top_k = top_k or config.TOP_K
        
        filter_dict = {"doc_id": {"$eq": doc_id}} if doc_id else None
        
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        chunks = []
        for match in results.matches:
            m = match.metadata
            headings_str = m.get("headings", "")
            chunks.append({
                "chunk_id": match.id,
                "text": m.get("text", ""),
                "score": match.score,
                "metadata": {
                    "source": m.get("source", ""),
                    "file_type": m.get("file_type", ""),
                    "chunk_index": m.get("chunk_index", 0),
                    "doc_id": m.get("doc_id", ""),
                    "headings": headings_str.split(" > ") if headings_str else [],
                    "pages": m.get("pages", []),
                    "chunk_type": m.get("chunk_type", "text"),
                }
            })
        
        logger.info(f"Pinecone search returned {len(chunks)} results")
        return chunks
        
    except Exception as e:
        logger.error(f"Pinecone search failed: {e}")
        raise


def delete_document(doc_id: str) -> bool:
    """
    Delete all chunks belonging to a document.
    Useful when re-uploading an updated policy.
    """
    try:
        pinecone_index = get_pinecone_index()
        
        pinecone_index.delete(
            filter={"doc_id": {"$eq": doc_id}}
        )
        
        logger.info(f"Deleted all chunks for doc_id: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Pinecone delete failed: {e}")
        raise
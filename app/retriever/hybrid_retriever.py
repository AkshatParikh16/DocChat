from app.retriever.pinecone_retriever import search_pinecone
from app.retriever.bm25_retriever import search_bm25
from app.embedder.embedder import embed_query
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)

# Cross-encoder reranker singleton — loaded once on first use
_reranker = None


def _get_reranker():
    """Load cross-encoder reranker (singleton, lazy)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
        _reranker = CrossEncoder(config.RERANKER_MODEL)
        logger.info("Reranker loaded")
    return _reranker


# ─── Score normalisation ───────────────────────────────────

def normalize_scores(results: list[dict]) -> list[dict]:
    """Min-max normalise scores to [0, 1] so Pinecone and BM25 are comparable."""
    if not results:
        return results

    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)

    if hi == lo:
        for r in results:
            r["normalized_score"] = 1.0
        return results

    for r in results:
        r["normalized_score"] = (r["score"] - lo) / (hi - lo)

    return results


# ─── Main entry point ──────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = None,
    doc_id: str = None,
    pinecone_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """
    Retrieve relevant chunks via Pinecone (semantic) + BM25 (keyword),
    merge by chunk_id with a weighted hybrid score, then optionally rerank
    with a cross-encoder for higher precision.

    Pipeline:
        1. Embed query → Pinecone ANN search
        2. BM25 keyword search
        3. Normalise + merge → hybrid score
        4. Cross-encoder reranker (if USE_RERANKER=true)
        5. Return top_k chunks with confidence signals
    """
    top_k = top_k or config.TOP_K
    candidate_k = top_k * 3  # fetch more candidates for reranker to prune

    logger.info(f"Hybrid search: '{query[:60]}'")

    # ── Step 1 & 2: parallel retrieval ────────────────────
    query_embedding = embed_query(query)

    pinecone_results = search_pinecone(
        query_embedding=query_embedding,
        top_k=candidate_k,
        doc_id=doc_id,
    )
    bm25_results = search_bm25(
        query=query,
        top_k=candidate_k,
        doc_id=doc_id,
    )

    # ── Step 3: normalise + merge ─────────────────────────
    pinecone_results = normalize_scores(pinecone_results)
    bm25_results = normalize_scores(bm25_results)

    merged: dict[str, dict] = {}

    for chunk in pinecone_results:
        cid = chunk["chunk_id"]
        merged[cid] = {
            "chunk_id": cid,
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "pinecone_score": chunk["normalized_score"],
            "bm25_score": 0.0,
            "raw_pinecone_score": chunk["score"],
        }

    for chunk in bm25_results:
        cid = chunk["chunk_id"]
        if cid in merged:
            merged[cid]["bm25_score"] = chunk["normalized_score"]
        else:
            merged[cid] = {
                "chunk_id": cid,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "pinecone_score": 0.0,
                "bm25_score": chunk["normalized_score"],
                "raw_pinecone_score": 0.0,
            }

    for chunk in merged.values():
        chunk["hybrid_score"] = (
            pinecone_weight * chunk["pinecone_score"]
            + bm25_weight * chunk["bm25_score"]
        )

    candidates = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)

    # ── Step 4: cross-encoder reranking ───────────────────
    if config.USE_RERANKER and candidates:
        candidates = _rerank(query, candidates, top_k)
    else:
        candidates = candidates[:top_k]

    # ── Step 5: attach confidence signals ─────────────────
    top_score = candidates[0]["hybrid_score"] if candidates else 0.0
    for chunk in candidates:
        chunk["confidence_signals"] = {
            "top_score": top_score,
            "found_in_both": chunk["pinecone_score"] > 0 and chunk["bm25_score"] > 0,
            "semantic_strong": chunk["pinecone_score"] > 0.7,
            "keyword_strong": chunk["bm25_score"] > 0.7,
        }

    logger.info(
        f"Hybrid search done: {len(candidates)} results | "
        f"top score: {top_score:.3f} | reranked: {config.USE_RERANKER}"
    )
    return candidates


# ─── Reranker ─────────────────────────────────────────────

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Use cross-encoder to re-score query↔chunk pairs and return top_k.

    The cross-encoder reads both query and passage together (unlike bi-encoders
    that encode them separately), giving it far better precision at the cost of
    being slower — acceptable here because we only score a small candidate pool.
    """
    try:
        reranker = _get_reranker()
        pairs = [(query, chunk["text"]) for chunk in candidates]
        scores = reranker.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk["reranker_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
        logger.info(
            f"Reranked {len(candidates)} → {top_k} | "
            f"top reranker score: {reranked[0]['reranker_score']:.3f}"
        )
        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Reranker failed ({e}), using hybrid scores instead")
        return candidates[:top_k]

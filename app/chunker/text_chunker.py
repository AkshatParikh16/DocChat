import uuid
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)


def chunk_document(document: dict) -> list[dict]:
    """
    Split a parsed document into chunks.

    Uses Docling's HybridChunker when a structured DoclingDocument is present
    (PDF, DOCX, PPTX, images) — it respects section boundaries, never splits
    mid-heading, and attaches the heading path to every chunk.

    Falls back to RecursiveCharacterTextSplitter for plain TXT/MD files.

    Image/figure entries from the parser are appended as their own chunks.

    Args:
        document: Output from load_document()

    Returns:
        List of chunk dicts, each with text + rich metadata
    """
    source = document["metadata"]["source"]
    docling_doc = document.get("docling_doc")

    if docling_doc is not None:
        text_chunks = _chunk_with_docling(docling_doc, document["metadata"])
    else:
        text_chunks = _chunk_with_splitter(document["text"], document["metadata"])

    image_chunks = _build_image_chunks(
        document.get("images", []), document["metadata"], base_index=len(text_chunks)
    )

    all_chunks = text_chunks + image_chunks

    logger.info(
        f"Chunked '{source}' → "
        f"{len(text_chunks)} text + {len(image_chunks)} image = {len(all_chunks)} total"
    )
    return all_chunks


# ─── Docling HybridChunker ─────────────────────────────────

def _chunk_with_docling(docling_doc, metadata: dict) -> list[dict]:
    """
    Structure-aware chunking via Docling HybridChunker.

    Each chunk is bounded by document sections (never straddles a heading),
    respects a token budget, and carries the full heading hierarchy in metadata
    so downstream retrieval knows *where* in the document each chunk lives.
    """
    try:
        from docling.chunking import HybridChunker

        chunker = HybridChunker(
            tokenizer=config.CHUNKER_TOKENIZER,
            max_tokens=config.CHUNK_TOKENS,
            merge_peers=True,
        )

        raw_chunks = list(chunker.chunk(docling_doc))
        chunks = []

        for i, chunk in enumerate(raw_chunks):
            text = chunk.text.strip()
            if not text:
                continue

            headings = list(getattr(chunk.meta, "headings", None) or [])
            pages = _extract_pages(chunk)

            chunks.append(_make_chunk(
                text=text,
                metadata=metadata,
                index=i,
                total=len(raw_chunks),
                headings=headings,
                pages=pages,
                chunk_type="text",
            ))

        return chunks

    except ImportError:
        logger.warning("docling.chunking not available — falling back to splitter")
        return _chunk_with_splitter(docling_doc.export_to_markdown(), metadata)
    except Exception as e:
        logger.warning(f"HybridChunker failed ({e}) — falling back to splitter")
        try:
            return _chunk_with_splitter(docling_doc.export_to_markdown(), metadata)
        except Exception:
            return _chunk_with_splitter("", metadata)


def _extract_pages(chunk) -> list[int]:
    """Pull unique page numbers from a HybridChunker chunk's provenance."""
    pages = set()
    try:
        for item in (chunk.meta.doc_items or []):
            for prov in (item.prov or []):
                pg = getattr(prov, "page_no", None)
                if pg is not None:
                    pages.add(pg)
    except Exception:
        pass
    return sorted(pages)


# ─── Fallback: RecursiveCharacterTextSplitter ─────────────

def _chunk_with_splitter(text: str, metadata: dict) -> list[dict]:
    """Character-based splitter for plain text files."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    return [
        _make_chunk(
            text=raw_chunks[i],
            metadata=metadata,
            index=i,
            total=len(raw_chunks),
            headings=[],
            pages=[],
            chunk_type="text",
        )
        for i in range(len(raw_chunks))
    ]


# ─── Image chunks ──────────────────────────────────────────

def _build_image_chunks(
    images: list[dict], metadata: dict, base_index: int = 0
) -> list[dict]:
    """Convert parser image objects into the standard chunk format."""
    chunks = []
    for i, img in enumerate(images):
        text = img.get("text", "").strip()
        if not text:
            continue
        chunks.append(_make_chunk(
            text=text,
            metadata=metadata,
            index=base_index + i,
            total=None,
            headings=[],
            pages=[img["page"]] if img.get("page") else [],
            chunk_type="image",
            extra={"figure_index": img.get("figure_index", i + 1)},
        ))
    return chunks


# ─── Shared chunk builder ──────────────────────────────────

def _make_chunk(
    text: str,
    metadata: dict,
    index: int,
    total: int | None,
    headings: list[str],
    pages: list[int],
    chunk_type: str,
    extra: dict | None = None,
) -> dict:
    chunk_meta = {
        "source": metadata["source"],
        "file_type": metadata["file_type"],
        "chunk_index": index,
        "total_chunks": total,
        "headings": headings,
        "pages": pages,
        "chunk_type": chunk_type,
    }
    if extra:
        chunk_meta.update(extra)

    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": metadata["source"],
        "chunk_index": index,
        "text": text,
        "metadata": chunk_meta,
    }

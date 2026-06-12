import os
from app.logger.logger import get_logger

logger = get_logger(__name__)

# Formats Docling handles natively (structural, layout-aware)
DOCLING_FORMATS = {".pdf", ".docx", ".pptx", ".html", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
# Plain text — no need for Docling overhead
PLAIN_TEXT_FORMATS = {".txt", ".md"}


def load_document(file_path: str) -> dict:
    """
    Parse a document and return a structured representation.

    Returns:
        {
            "text":        str              — full markdown text of the document
            "metadata":    dict             — source, file_type, total_pages
            "docling_doc": DoclingDocument  — structured doc for HybridChunker (None for TXT)
            "images":      list[dict]       — figures with caption text (empty for TXT)
        }
    """
    ext = os.path.splitext(file_path)[1].lower()
    source_name = os.path.basename(file_path)

    if ext in PLAIN_TEXT_FORMATS:
        return _load_plain_text(file_path, source_name, ext)
    elif ext in DOCLING_FORMATS:
        return _load_with_docling(file_path, source_name, ext)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: PDF, DOCX, PPTX, TXT, MD, PNG, JPG, JPEG, TIFF"
        )


# ─── Plain text ────────────────────────────────────────────

def _load_plain_text(file_path: str, source_name: str, ext: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        logger.info(f"Plain text loaded: {source_name}")
        return {
            "text": text.strip(),
            "metadata": {
                "source": source_name,
                "file_type": ext.lstrip("."),
                "total_pages": None
            },
            "docling_doc": None,
            "images": []
        }
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


# ─── Docling (PDF / DOCX / PPTX / images) ─────────────────

def _load_with_docling(file_path: str, source_name: str, ext: str) -> dict:
    """
    Use Docling to parse documents with full structural understanding:
    headings, paragraphs, tables (→ markdown), figures (→ caption text).

    On first run Docling downloads layout + OCR models (~450 MB total) to
    ~/.cache/docling/models/ — cached permanently after that.
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        # For PDFs: enable picture metadata extraction
        if ext == ".pdf":
            pdf_opts = PdfPipelineOptions()
            pdf_opts.generate_picture_images = True
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
            )
        else:
            converter = DocumentConverter()

        logger.info(f"Docling parsing: {source_name}")
        result = converter.convert(file_path)
        doc = result.document

        # Export to markdown — preserves headings (#, ##), tables, lists
        markdown_text = doc.export_to_markdown()

        # Count pages (PDF only)
        total_pages = len(doc.pages) if hasattr(doc, "pages") and doc.pages else None

        # Extract image/figure chunks
        image_chunks = _extract_image_chunks(doc, source_name)

        logger.info(
            f"Docling done: {source_name} | "
            f"pages={total_pages} | figures={len(image_chunks)}"
        )

        return {
            "text": markdown_text.strip(),
            "metadata": {
                "source": source_name,
                "file_type": ext.lstrip("."),
                "total_pages": total_pages
            },
            "docling_doc": doc,
            "images": image_chunks
        }

    except ImportError:
        logger.error("Docling not installed. Run: pip install docling")
        raise
    except Exception as e:
        logger.error(f"Docling failed for {file_path}: {e}")
        raise


# ─── Image / figure extraction ────────────────────────────

def _extract_image_chunks(doc, source_name: str) -> list[dict]:
    """
    Turn figures with captions into searchable text snippets.
    Images without any caption are skipped (no useful text to embed).
    """
    image_chunks = []

    try:
        for i, picture in enumerate(doc.pictures):
            caption_text = _get_caption_text(picture)
            if not caption_text:
                continue

            page_no = _get_page_no(picture)

            label = f"[Figure {i + 1}"
            if page_no:
                label += f", Page {page_no}"
            label += "]"

            image_chunks.append({
                "text": f"{label}\n{caption_text}",
                "page": page_no,
                "figure_index": i + 1
            })

    except Exception as e:
        logger.warning(f"Image extraction warning ({source_name}): {e}")

    return image_chunks


def _get_caption_text(picture) -> str:
    """Safely pull caption text from a Docling PictureItem."""
    if not hasattr(picture, "captions") or not picture.captions:
        return ""
    parts = []
    for cap in picture.captions:
        if hasattr(cap, "text") and cap.text:
            parts.append(cap.text)
        elif isinstance(cap, str) and cap:
            parts.append(cap)
    return " ".join(parts).strip()


def _get_page_no(picture) -> int | None:
    """Safely pull page number from a Docling PictureItem's provenance."""
    try:
        if hasattr(picture, "prov") and picture.prov:
            return getattr(picture.prov[0], "page_no", None)
    except Exception:
        pass
    return None

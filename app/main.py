from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import shutil
from app.pipeline import ingest_document, answer_query, delete_doc, compare_policies
from app.llm import get_llm
from app.logger.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="PolicyChat - Insurance Policy Intelligence",
    description="RAG-based insurance policy analysis system",
    version="2.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ───────────────────────────────

class QueryRequest(BaseModel):
    query: str
    doc_id: str = None


class CompareRequest(BaseModel):
    query: str
    doc_id_1: str
    doc_id_2: str


class DeleteRequest(BaseModel):
    doc_id: str


# ─── Root ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "PolicyChat",
        "version": "2.0.0",
        "endpoints": {
            "GET  /health":      "Check system status",
            "GET  /documents":   "List indexed documents",
            "POST /ingest":      "Upload policy document",
            "POST /query":       "Ask question about policies",
            "POST /compare":     "Compare two policies",
            "DELETE /document":  "Remove a document"
        }
    }


# ─── Health Check ──────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if API and Ollama are running."""
    llm = get_llm()
    ollama_status = llm.is_available()
    return {
        "status": "ok",
        "ollama": "running" if ollama_status else "not running",
        "message": "PolicyChat API is ready" if ollama_status else "Start Ollama: ollama serve"
    }


# ─── List Documents ────────────────────────────────────────

@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    try:
        from app.retriever.bm25_retriever import stored_chunks
        docs = list(set([
            chunk["metadata"]["source"]
            for chunk in stored_chunks
        ]))
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Document Ingestion ────────────────────────────────────

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload and index a policy document."""
    allowed = [".pdf", ".docx", ".pptx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: PDF, DOCX, PPTX, TXT, MD, PNG, JPG, JPEG, TIFF"
        )

    # Save with REAL filename preserved
    tmp_dir = tempfile.mkdtemp()
    real_filename = file.filename.replace(" ", "_")
    tmp_path = os.path.join(tmp_dir, real_filename)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        result = ingest_document(tmp_path)
        result["filename"] = real_filename
        return result

    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── Query ─────────────────────────────────────────────────

@app.post("/query")
def query(request: QueryRequest):
    """Ask a question about uploaded policy documents."""
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    try:
        result = answer_query(
            query=request.query,
            doc_id=request.doc_id
        )
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Compare ───────────────────────────────────────────────

@app.post("/compare")
def compare(request: CompareRequest):
    """Compare two policy documents on a specific query."""
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    try:
        result = compare_policies(
            query=request.query,
            doc_id_1=request.doc_id_1,
            doc_id_2=request.doc_id_2
        )
        return result
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Delete Document ───────────────────────────────────────

@app.delete("/document")
def delete_document(request: DeleteRequest):
    """Remove a document from the system."""
    try:
        result = delete_doc(request.doc_id)
        return result
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
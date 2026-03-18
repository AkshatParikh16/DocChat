<h1 align="center">
  <br>
  🤖 DocChat
  <br>
</h1>

<h4 align="center">AI-powered document intelligence. Ask anything. Get grounded answers — directly from your documents.</h4>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LLaMA3-Ollama-black?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Pinecone-Vector_DB-6C2DC7?style=for-the-badge" />
</p>

<p align="center">
  <a href="#-demo">Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#️-how-it-works">How It Works</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-api">API</a> •
  <a href="#️-tech-stack">Tech Stack</a>
</p>

---

## What is DocChat?

DocChat is an end-to-end **RAG (Retrieval-Augmented Generation)** system for intelligent document analysis. Upload any PDF, ask questions in plain English, and get precise answers pulled directly from your document — with source citations.

It never guesses. If the answer isn't in your document, it tells you that.

---

## 🎬 Demo

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/ss1.png" width="100%" />
      <br><sub><b>Landing Page</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="assets/ss2.png" width="100%" />
      <br><sub><b>Platform Modules</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/ss3.png" width="100%" />
      <br><sub><b>Document Analysis</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="assets/ss4.png" width="100%" />
      <br><sub><b>Document Comparison</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/ss5.png" width="100%" />
      <br><sub><b>Multi-Document Chat</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="assets/ss6.png" width="100%" />
      <br><sub><b>Architecture</b></sub>
    </td>
  </tr>
</table>

---

## ✨ Features

- 📄 **Document Analysis** — Upload any PDF and ask questions. Answers are grounded in the document with source citations.
- ⚖️ **Document Comparison** — Compare multiple documents side by side. Full document isolation — answers never cross boundaries.
- 💬 **Multi-Document Chat** — Conversational interface across your entire document library with per-message scope control.
- 🔒 **Hallucination Prevention** — Strict confidence scoring. If retrieval is weak, the LLM never gets called.
- 🛡️ **Safety Layer** — Automatically detects high-risk queries and adds appropriate disclaimers.
- 🏠 **Fully Local** — LLM runs on your machine via Ollama. Documents never leave your device.

---

## ⚙️ How It Works

```
Your Document
   │
   ├─► Parse → Chunk (1200 chars) → Embed (all-MiniLM-L6-v2)
   │                                        │
   │                               ┌────────┴────────┐
   │                            Pinecone           BM25
   │                          (semantic)         (keyword)
   │
Your Question
   │
   ├─► Hybrid Retrieval (0.7 semantic + 0.3 keyword)
   │
   ├─► Safety Check + Confidence Scoring
   │
   └─► LLaMA3 → Grounded Answer with Citations
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed
- Pinecone account (free tier works)

### Installation

```bash
# Clone the repo
git clone https://github.com/AkshatParikh16/PolicyChat.git
cd PolicyChat

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the root:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct-q4_0
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=policychat
CONFIDENCE_THRESHOLD=0.35
TOP_K=5
```

> Pinecone index settings: **384 dimensions**, **cosine** metric, **Serverless**

### Run

```bash
# Terminal 1 — Start Ollama
ollama pull llama3:8b-instruct-q4_0
ollama serve

# Terminal 2 — Start API
python run.py
# → http://localhost:8000

# Terminal 3 — Start Frontend
python policychat/serve.py
# → http://localhost:3000
```

---

## 📡 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System + Ollama status |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/ingest` | Upload a document |
| `POST` | `/query` | Ask a question |
| `POST` | `/compare` | Compare two documents |
| `DELETE` | `/document` | Remove a document |

Interactive docs at **http://localhost:8000/docs**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| LLM | LLaMA3 8B Q4 (local via Ollama) |
| Embeddings | `all-MiniLM-L6-v2` — HuggingFace (local) |
| Vector DB | Pinecone Serverless |
| Keyword Search | BM25 (Okapi variant) |
| Document Parsing | PyMuPDF, python-docx |
| Frontend | HTML / CSS |

---

## 📁 Project Structure

```
DocChat/
├── app/
│   ├── main.py               # FastAPI routes
│   ├── pipeline.py           # Core orchestration
│   ├── config.py             # Environment config
│   ├── parser/               # PDF/DOCX/TXT extraction
│   ├── chunker/              # Text splitting
│   ├── embedder/             # Vector generation
│   ├── retriever/            # Pinecone + BM25 + hybrid
│   ├── safety/               # Risk + confidence scoring
│   └── llm/                  # LLM provider pattern
├── policychat/
│   ├── index.html            # Frontend SPA
│   └── serve.py              # Static file server
├── assets/                   # Screenshots
├── run.py
└── requirements.txt
```

---

<p align="center">Built by <strong>Akshat Parikh</strong> · MS Data Science, UT Arlington</p>
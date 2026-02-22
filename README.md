# RAG Embedding Comparison

A benchmarking system for comparing two embedding model sizes (4B and 8B) on RAG (Retrieval-Augmented Generation) tasks over Thai-language PDF documents. Answer quality is evaluated using an LLM Judge, and OCR accuracy is measured with WER (Word Error Rate).

---

## Objectives

- **Compare Embedding Models**: Measure whether `qwen3-embedding:4b` or `qwen3-embedding:8b` produces better RAG results on Thai-language documents.
- **Compare Chunking Strategies**: Test both Recursive Character Split and Agentic (LLM-based semantic) Chunking.
- **Evaluate with LLM Judge**: Use an LLM to score answers 0–100 against a Golden Answer.
- **Measure OCR Quality**: Calculate WER against Ground Truth to assess the quality of OCR-extracted text.

---

## System Architecture

```
PDF Upload
    │
    ▼
OCR (Qwen3-VL via Ollama)
    │
    ▼
Chunking ──────────────────────────────────┐
  ├── Recursive (LangChain splitter)       │
  └── Agentic (LLM semantic chunking)     │
    │                                      │
    ▼                                      │
Embedding Generation                       │
  ├── Model 4B (qwen3-embedding:4b-q8_0)  │
  └── Model 8B (qwen3-embedding:8b)       │
    │                                      │
    ▼                                      │
RAG Pipeline ◄─────────────────────────────┘
  ├── Cosine Similarity Retrieval (Top-K)
  └── LLM Answer Generation
    │
    ▼
LLM Evaluation (Judge)
  ├── Score 4B (0–100)
  └── Score 8B (0–100)
    │
    ▼
WER Assessment (OCR Quality)
```

---

## Project Structure

```
embedding evalute/
├── app.py                  # FastAPI application + REST API endpoints
├── config.py               # All configuration (DB, Ollama, models, parameters)
├── database.py             # SQLAlchemy models + DB initialization
├── ocr_service.py          # Convert PDF to images and run OCR with VLM
├── chunking_service.py     # Recursive and Agentic chunking
├── embedding_service.py    # Embedding generation + cosine similarity retrieval
├── rag_service.py          # RAG pipeline (retrieve + LLM answer generation)
├── evaluation_service.py   # LLM Judge for scoring answers
├── wer_service.py          # OCR Word Error Rate computation
├── evaluation_prompt.txt   # Evaluation prompt template (editable from UI)
├── requirements.txt        # Python dependencies
├── templates/              # Jinja2 HTML templates
├── static/                 # CSS, JS
├── uploads/                # Uploaded PDF files and page images
└── best_ocr/               # Ground truth OCR text (page_1.txt, page_2.txt, ...)
```

---

## Module Details

### `app.py` — FastAPI Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main Web UI page |
| `/api/upload` | POST | Upload PDF and run OCR |
| `/api/chunk/{session_id}` | POST | Create chunks (recursive + agentic) |
| `/api/embed/{session_id}` | POST | Generate embeddings with both models |
| `/api/questions/{session_id}` | GET/POST | Manage Questions + Golden Answers |
| `/api/rag/{session_id}` | POST | Run RAG pipeline |
| `/api/evaluate/{session_id}` | POST | Evaluate answers with LLM Judge |
| `/api/wer/{session_id}` | POST | Compute WER for OCR pages |
| `/api/prompt/evaluation` | GET/POST | View / edit evaluation prompt |
| `/api/sessions` | GET | List all sessions |
| `/api/results/{session_id}` | GET | View all results for a session |
| `/api/llm-config` | GET | View current LLM parameter defaults |

---

### `database.py` — Database Schema

| Table | Description |
|---|---|
| `upload_sessions` | One session per uploaded PDF, with status tracking the workflow |
| `ocr_pages` | OCR result for each page, including image path and WER score |
| `recursive_chunks` | Chunks from LangChain `RecursiveCharacterTextSplitter` |
| `agentic_chunks` | Chunks from LLM semantic chunking (each has a title) |
| `embeddings_4b` | Embedding vectors (float32 bytes) from the 4B model |
| `embeddings_8b` | Embedding vectors (float32 bytes) from the 8B model |
| `questions` | Questions and Golden Answers for each session |
| `evaluation_results` | RAG answers + LLM Judge scores |
| `wer_results` | OCR vs. ground truth comparison results |

**Session Status Flow:**
```
uploaded → ocr_done → chunked → embedded → rag_done → evaluated
```

---

### `chunking_service.py` — Chunking Strategies

**Recursive Chunking**
- Uses `LangChain RecursiveCharacterTextSplitter`
- Chunk size: 1,300 characters, Overlap: 30 characters
- Splits on separators: `\n\n`, `\n`, `。`, `.`, ` `

**Agentic Chunking**
- Sends each page's text to an LLM to split into semantic chunks
- Each chunk is given a descriptive `title`
- Falls back to treating the whole page as one chunk if the LLM fails

---

### `embedding_service.py` — Embedding & Retrieval

- Calls the Ollama `/api/embed` endpoint to generate embedding vectors
- Computes **Cosine Similarity** to retrieve the top-K chunks most relevant to the query
- Supports unloading the model after use to free VRAM

---

### `evaluation_service.py` — LLM Judge

- Sends the question, Golden Answer, and both model answers to an LLM for scoring
- Extracts `SCORE_4B` and `SCORE_8B` from the response using regex
- Supports a custom prompt template via `evaluation_prompt.txt`

**Required placeholders in the evaluation prompt:**
```
{question}, {golden_answer}, {answer_4b}, {answer_8b}
```

---

### `wer_service.py` — OCR Quality Assessment

- Compares OCR text against ground truth files in the `best_ocr/` directory
- Normalizes text (strips markdown, collapses whitespace) before computing
- Uses the `jiwer` library for WER calculation
- WER = 0 is perfect; WER > 1 indicates heavy errors

---

## Configuration

### Create a `.env` file

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=test_embedding
DB_USER=myuser
DB_PASSWORD=mypassword

# Ollama
OLLAMA_BASE_URL=http://localhost:7869

# Models
OCR_MODEL=qwen3-vl:30b-a3b-instruct-q8_0
LLM_MODEL=gpt-oss:120b
EMBEDDING_MODEL_4B=qwen3-embedding:4b-q8_0
EMBEDDING_DIM_4B=2560
EMBEDDING_MODEL_8B=qwen3-embedding:8b
EMBEDDING_DIM_8B=4096

# LLM Parameters
LLM_TEMPERATURE=0.6
LLM_TOP_P=0.95
LLM_MAX_PREDICT=25000
LLM_NUM_CTX=50000

# Chunking
RECURSIVE_CHUNK_SIZE=1300
RECURSIVE_CHUNK_OVERLAP=30

# RAG
RAG_TOP_K=5

# App
APP_HOST=0.0.0.0
APP_PORT=8000
```

---

## Installation & Running

### 1. Prepare the Environment

```bash
conda activate py131
cd "/home/indows-11/my_code/VENV"
source venv_qwen3vl/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare PostgreSQL

```bash
# Create the database (tables are created automatically on first run)
createdb test_embedding
```

### 4. Run the Application

```bash
cd "/home/indows-11/my_code/claude code/embedding evalute"
python app.py
```

Open a browser at `http://localhost:8000`

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.115.0 | Web framework |
| `uvicorn` | 0.30.6 | ASGI server |
| `sqlalchemy` | 2.0.35 | ORM + PostgreSQL |
| `psycopg2-binary` | 2.9.9 | PostgreSQL driver |
| `httpx` | 0.27.2 | Ollama API client |
| `langchain-text-splitters` | 0.3.0 | Recursive chunking |
| `jiwer` | 3.0.4 | Word Error Rate |
| `PyMuPDF` | 1.24.10 | PDF to image conversion |
| `Pillow` | 10.4.0 | Image processing |
| `numpy` | 1.26.4 | Vector operations |

---

## Required Ollama Models

```bash
ollama pull qwen3-vl:30b-a3b-instruct-q8_0   # OCR / LLM Judge
ollama pull qwen3-embedding:4b-q8_0            # Embedding 4B
ollama pull qwen3-embedding:8b                 # Embedding 8B
```

---

## Ground Truth for WER

Place reference text files in `best_ocr/`, named by page number:

```
best_ocr/
├── page_1.txt
├── page_2.txt
└── page_3.txt
```

---

## Usage Workflow

```
1. Upload PDF       → OCR runs automatically
2. Create Chunks    → Choose Recursive / Agentic / Both
3. Create Embeddings → Generates vectors with both 4B and 8B models
4. Enter Questions + Golden Answers
5. Run RAG          → System retrieves context and generates answers
6. Evaluate         → LLM scores SCORE_4B and SCORE_8B
7. (Optional) Compute WER → Inspect OCR quality per page
```

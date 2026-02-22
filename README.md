# Thai Banking Compliance — RAG Embedding Benchmark

> A benchmarking system for comparing multiple embedding model sizes (**0.6B, 4B, 8B** Qwen3 and **BGE-M3**) on RAG (Retrieval-Augmented Generation) tasks over Thai-language PDF documents. Answer quality is evaluated using an LLM Judge, and OCR accuracy is measured with WER (Word Error Rate).

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Screenshots

### Evaluation Results — LLM Judge Scoring

<p align="center">
  <img src="Image/Screenshot 2026-02-22 123643.png" alt="Evaluation Results" width="900"/>
</p>

> **แท็บ Evaluation** แสดงผลการให้คะแนนของ LLM Judge สำหรับแต่ละ Embedding Model (4B, 8B, BGE-M3) โดยแต่ละคำถามจะแสดง Score Card เปรียบเทียบคะแนน 0–100 ระหว่างโมเดล พร้อมคำตอบที่ถูก Retrieve มาจากแต่ละ Model แบบเคียงกัน เพื่อให้เห็นความแตกต่างของคุณภาพคำตอบได้ชัดเจน

---

### RAG Results — Retrieved Chunks Comparison

<p align="center">
  <img src="Image/Screenshot 2026-02-22 123740.png" alt="RAG Results" width="900"/>
</p>

> **แท็บ RAG Results** แสดง Chunk ที่ถูก Retrieve มาจากแต่ละ Embedding Model เคียงกันในแต่ละคำถาม ช่วยให้เปรียบเทียบได้ว่าโมเดลไหน Retrieve Chunk ที่เกี่ยวข้องได้ตรงประเด็นมากกว่า รวมถึงแสดง Similarity Score และ Chunk Index ของแต่ละผลลัพธ์

---

### WER Assessment — OCR Quality

<p align="center">
  <img src="Image/Screenshot 2026-02-22 123748.png" alt="WER Assessment" width="900"/>
</p>

> **แท็บ WER** แสดงผลการวัดคุณภาพ OCR โดยใช้ Word Error Rate (WER) เปรียบเทียบ OCR Output ของแต่ละหน้ากับ Reference Text ใน `best_ocr/` ตัวอย่างนี้แสดง WER รวม **24.8%** จากทั้งหมด 16 หน้า พร้อม Diff แบบ Inline ระหว่าง OCR Output (ซ้าย) และ Reference Best OCR (ขวา)

---

### Evaluation Prompt Editor

<p align="center">
  <img src="Image/Screenshot 2026-02-22 123757.png" alt="Evaluation Prompt Editor" width="700"/>
</p>

> **หน้าต่าง Evaluation Prompt** ให้ผู้ใช้แก้ไข Prompt ที่ใช้ส่งให้ LLM Judge ได้โดยตรงผ่าน UI โดยไม่ต้องแก้ไขโค้ด รองรับ Placeholder ได้แก่ `{question}`, `{golden_answer}`, `{answer_4b}`, `{answer_8b}`, `{answer_bge}` และกำหนด Output Format ให้ LLM ตอบกลับเป็น `SCORE_4B`, `SCORE_8B`, `SCORE_BGE` และ `SCORE_8GNT`

---

### Golden Dataset — Questions & Answers

<p align="center">
  <img src="Image/Screenshot 2026-02-22 123807.png" alt="Golden Dataset" width="700"/>
</p>

> **หน้าต่าง Golden Dataset** ใช้สำหรับจัดการชุดคำถามและ Golden Answer (คำตอบอ้างอิง) ที่ใช้ประเมินผล LLM Judge จะนำ Golden Answer นี้ไปเปรียบเทียบกับคำตอบของแต่ละ Embedding Model เพื่อให้คะแนน ตัวอย่างนี้มี 3 คำถามเกี่ยวกับ Digital Fraud Management และ Compliance สำหรับธนาคาร

---

## Objectives

- **Compare Embedding Models**: Measure whether `qwen3-embedding:0.6b`, `qwen3-embedding:4b`, `qwen3-embedding:8b`, or `bge-m3` produces better RAG results on Thai-language documents.
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
  ├── Model 0.6B (qwen3-embedding:0.6b)   │
  ├── Model 4B   (qwen3-embedding:4b-q8_0)│
  ├── Model 8B   (qwen3-embedding:8b)     │
  └── BGE-M3     (bge-m3)                 │
    │                                      │
    ▼                                      │
RAG Pipeline ◄─────────────────────────────┘
  ├── Cosine Similarity Retrieval (Top-K)
  └── LLM Answer Generation
    │
    ▼
LLM Evaluation (Judge)
  ├── SCORE_4B   (0–100)
  ├── SCORE_8B   (0–100)
  ├── SCORE_BGE  (0–100)
  └── SCORE_8GNT (0–100)
    │
    ▼
WER Assessment (OCR Quality)
```

---

## Project Structure

```
compare_qwen3_embedding/
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
├── Image/                  # Screenshots and documentation images
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
| `/api/embed/{session_id}` | POST | Generate embeddings with all models |
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

- Sends the question, Golden Answer, and all model answers to an LLM for scoring
- Extracts `SCORE_4B`, `SCORE_8B`, `SCORE_BGE` from the response using regex
- Supports a custom prompt template via `evaluation_prompt.txt`

**Required placeholders in the evaluation prompt:**
```
{question}, {golden_answer}, {answer_4b}, {answer_8b}, {answer_bge}
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
cd "/home/indows-11/my_code/claude code/compare_qwen3_embedding_0.6b_4b_8b_bgm-m3-"
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
ollama pull qwen3-embedding:0.6b               # Embedding 0.6B
ollama pull qwen3-embedding:4b-q8_0            # Embedding 4B
ollama pull qwen3-embedding:8b                 # Embedding 8B
ollama pull bge-m3                             # Embedding BGE-M3
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
1. Upload PDF         → OCR runs automatically
2. Create Chunks      → Choose Recursive / Agentic / Both
3. Create Embeddings  → Generates vectors with all models (0.6B, 4B, 8B, BGE-M3)
4. Enter Questions + Golden Answers
5. Run RAG            → System retrieves context and generates answers
6. Evaluate           → LLM scores SCORE_4B, SCORE_8B, SCORE_BGE
7. (Optional) Compute WER → Inspect OCR quality per page
```

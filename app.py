import os
import json
import shutil
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional

from config import (
    APP_HOST, APP_PORT, UPLOAD_DIR, BEST_OCR_DIR,
    EMBEDDING_MODELS,
    EMBEDDING_MODEL_4B, EMBEDDING_MODEL_8B,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_PREDICT, LLM_NUM_CTX,
    apply_llm_overrides,
)
from database import (
    init_db, SessionLocal, UploadSession, OcrPage,
    RecursiveChunk, AgenticChunk,
    ChunkEmbedding,
    Question, EvaluationResult, WerResult,
    emb_to_db, emb_from_db,
)
from ocr_service import process_pdf
from chunking_service import create_recursive_chunks, create_agentic_chunks
from embedding_service import get_embeddings_batch, unload_model
from rag_service import run_rag_pipeline
from evaluation_service import (
    evaluate_all, get_evaluation_prompt,
    DEFAULT_EVALUATION_PROMPT, PROMPT_FILE,
    REQUIRED_PLACEHOLDERS,
)
from wer_service import compute_wer_for_session


def _parse_retrieved_chunks(retrieved_chunks_json) -> list:
    """Parse retrieved_chunks JSON string from EvaluationResult into list of dicts."""
    if not retrieved_chunks_json:
        return []
    try:
        raw = json.loads(retrieved_chunks_json)
        result = []
        for item in raw:
            if isinstance(item, dict):
                result.append({
                    "text": item.get("text", ""),
                    "similarity": item.get("sim", 0),
                    "chunk_type": item.get("type", ""),
                })
            else:
                result.append({"text": str(item), "similarity": 0, "chunk_type": ""})
        return result
    except Exception:
        return []

app = FastAPI(title="Thai Banking Compliance Embedding Benchmark")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads_files")

# Mapping from model key → ChunkEmbedding column attribute name
_MODEL_EMB_COL = {
    "06b":   "embedding_06b",
    "4b":    "embedding_4b",
    "8b":    "embedding_8b",
    "bgem3": "embedding_bgem3",
}


def _image_url(image_path: str) -> str:
    """Convert filesystem image path to a /uploads/... URL."""
    if not image_path:
        return ""
    try:
        upload_dir_abs = os.path.abspath(UPLOAD_DIR)
        image_path_abs = os.path.abspath(image_path)
        rel = os.path.relpath(image_path_abs, upload_dir_abs)
        return "/uploads/" + rel.replace(os.sep, "/")
    except Exception:
        return ""


# Initialize DB on startup
@app.on_event("startup")
def startup():
    init_db()


# ==================== Pages ====================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ==================== Upload & OCR ====================

@app.post("/api/upload")
async def upload_pdfs(files: list[UploadFile] = File(...), override_session_id: int = Form(None)):
    """Upload PDF files and perform OCR on each page. Supports overriding existing sessions."""
    db = SessionLocal()
    results = []

    try:
        for file in files:
            if override_session_id:
                session = db.query(UploadSession).filter(UploadSession.id == override_session_id).first()
                if session:
                    print(f"\n🔄 Override: clearing old data for session {override_session_id}")
                    db.query(OcrPage).filter(OcrPage.session_id == override_session_id).delete()
                    db.query(RecursiveChunk).filter(RecursiveChunk.session_id == override_session_id).delete()
                    db.query(AgenticChunk).filter(AgenticChunk.session_id == override_session_id).delete()
                    db.query(ChunkEmbedding).filter(ChunkEmbedding.session_id == override_session_id).delete()
                    db.query(EvaluationResult).filter(EvaluationResult.session_id == override_session_id).delete()
                    db.query(Question).filter(Question.session_id == override_session_id).delete()
                    session.filename = file.filename
                    session.status = "uploading"
                    db.commit()
                else:
                    session = UploadSession(filename=file.filename, status="uploading")
                    db.add(session)
                    db.commit()
                    db.refresh(session)
            else:
                session = UploadSession(filename=file.filename, status="uploading")
                db.add(session)
                db.commit()
                db.refresh(session)

            session_dir = os.path.join(UPLOAD_DIR, f"session_{session.id}")
            os.makedirs(session_dir, exist_ok=True)
            pdf_path = os.path.join(session_dir, file.filename)

            with open(pdf_path, "wb") as f:
                content = await file.read()
                f.write(content)

            print(f"\n📄 Processing: {file.filename} (Session {session.id})")
            ocr_results = process_pdf(pdf_path, session.id)

            for page_result in ocr_results:
                ocr_page = OcrPage(
                    session_id=session.id,
                    page_number=page_result["page_number"],
                    image_path=page_result["image_path"],
                    ocr_text=page_result["ocr_text"],
                )
                db.add(ocr_page)

            session.total_pages = len(ocr_results)
            session.status = "ocr_done"
            db.commit()

            results.append({
                "session_id": session.id,
                "filename": file.filename,
                "total_pages": len(ocr_results),
                "status": "ocr_done",
                "pages": [
                    {
                        "page_number": r["page_number"],
                        "text_preview": r["ocr_text"][:200] if r["ocr_text"] else "",
                    }
                    for r in ocr_results
                ],
            })

        return JSONResponse({"status": "success", "results": results})

    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== Session Management ====================

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: int):
    """Delete a session and all related records and uploaded files."""
    db = SessionLocal()
    try:
        session = db.query(UploadSession).filter(UploadSession.id == session_id).first()
        if not session:
            return JSONResponse({"status": "error", "message": "Session not found"}, status_code=404)

        print(f"\n🗑️  Deleting session {session_id} and all related data...")
        db.query(OcrPage).filter(OcrPage.session_id == session_id).delete()
        db.query(RecursiveChunk).filter(RecursiveChunk.session_id == session_id).delete()
        db.query(AgenticChunk).filter(AgenticChunk.session_id == session_id).delete()
        db.query(ChunkEmbedding).filter(ChunkEmbedding.session_id == session_id).delete()
        db.query(Question).filter(Question.session_id == session_id).delete()
        db.query(EvaluationResult).filter(EvaluationResult.session_id == session_id).delete()
        db.query(WerResult).filter(WerResult.session_id == session_id).delete()
        db.delete(session)
        db.commit()

        # Delete uploaded files
        session_dir = os.path.join(UPLOAD_DIR, f"session_{session_id}")
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            print(f"  🗂️  Deleted directory: {session_dir}")

        return JSONResponse({"status": "success", "message": f"Session {session_id} deleted"})

    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== Chunking ====================

@app.post("/api/chunk/{session_id}")
async def create_chunks(session_id: int, request: Request):
    """Create recursive and agentic chunks for a session."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    apply_llm_overrides(body.get("llm_params", {}))

    db = SessionLocal()
    try:
        pages = db.query(OcrPage).filter(OcrPage.session_id == session_id).order_by(OcrPage.page_number).all()
        if not pages:
            return JSONResponse({"status": "error", "message": "No OCR pages found"}, status_code=404)

        page_data = [{"page_number": p.page_number, "ocr_text": p.ocr_text} for p in pages]

        print(f"\n🗑️  Clearing existing chunks for session {session_id}...")
        db.query(RecursiveChunk).filter(RecursiveChunk.session_id == session_id).delete()
        db.query(AgenticChunk).filter(AgenticChunk.session_id == session_id).delete()
        db.commit()

        print(f"\n📦 Creating recursive chunks for session {session_id}...")
        rec_chunks = create_recursive_chunks(page_data)
        for c in rec_chunks:
            db.add(RecursiveChunk(
                session_id=session_id,
                page_number=c["page_number"],
                chunk_index=c["chunk_index"],
                chunk_text=c["chunk_text"],
                chunk_size=c["chunk_size"],
            ))

        print(f"\n🧠 Creating agentic chunks for session {session_id}...")
        ag_chunks = create_agentic_chunks(page_data)
        for c in ag_chunks:
            db.add(AgenticChunk(
                session_id=session_id,
                page_number=c["page_number"],
                chunk_index=c["chunk_index"],
                chunk_text=c["chunk_text"],
                chunk_title=c.get("chunk_title", ""),
                chunk_size=c["chunk_size"],
            ))

        session = db.query(UploadSession).filter(UploadSession.id == session_id).first()
        if session:
            session.status = "chunked"
        db.commit()

        return JSONResponse({
            "status": "success",
            "recursive_chunks": len(rec_chunks),
            "agentic_chunks": len(ag_chunks),
        })

    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== Embeddings ====================

@app.post("/api/embed/{session_id}")
async def create_embeddings(session_id: int):
    """Generate embeddings for all chunks using all 4 embedding models."""
    db = SessionLocal()
    try:
        rec_chunks = db.query(RecursiveChunk).filter(
            RecursiveChunk.session_id == session_id
        ).order_by(RecursiveChunk.id).all()
        ag_chunks = db.query(AgenticChunk).filter(
            AgenticChunk.session_id == session_id
        ).order_by(AgenticChunk.id).all()

        all_chunks = []
        for c in rec_chunks:
            all_chunks.append({"id": c.id, "text": c.chunk_text, "type": "recursive"})
        for c in ag_chunks:
            all_chunks.append({"id": c.id, "text": c.chunk_text, "type": "agentic"})

        if not all_chunks:
            return JSONResponse({"status": "error", "message": "No chunks found"}, status_code=404)

        # Clear existing embeddings for this session
        print(f"\n🗑️  Clearing existing embeddings for session {session_id}...")
        db.query(ChunkEmbedding).filter(ChunkEmbedding.session_id == session_id).delete()
        db.commit()

        texts = [c["text"] for c in all_chunks]
        emb_counts = {}

        # Step 1: Insert one row per chunk (embeddings filled in next step)
        for chunk in all_chunks:
            db.add(ChunkEmbedding(
                session_id=session_id,
                chunk_id=chunk["id"],
                chunk_type=chunk["type"],
                chunk_text=chunk["text"],
            ))
        db.commit()

        # Retrieve inserted rows in order
        emb_rows = db.query(ChunkEmbedding).filter(
            ChunkEmbedding.session_id == session_id
        ).order_by(ChunkEmbedding.id).all()

        # Step 2: For each model, generate embeddings and update the corresponding column
        for key, ollama_model, label in EMBEDDING_MODELS:
            print(f"\n{label} Generating embeddings for {len(texts)} chunks...")
            embs = get_embeddings_batch(texts, ollama_model, unload_after=True)
            col_name = _MODEL_EMB_COL[key]
            for row, emb in zip(emb_rows, embs):
                setattr(row, col_name, emb_to_db(emb))
            db.commit()
            emb_counts[key] = len(embs)
            print(f"    ✅ {label}: {len(embs)} embeddings stored")

        session = db.query(UploadSession).filter(UploadSession.id == session_id).first()
        if session:
            session.status = "embedded"
        db.commit()

        return JSONResponse({
            "status": "success",
            "total_chunks": len(all_chunks),
            **{f"embeddings_{k}": v for k, v in emb_counts.items()},
        })

    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== Questions ====================

@app.get("/api/questions/{session_id}")
async def get_questions(session_id: int):
    db = SessionLocal()
    try:
        questions = db.query(Question).filter(
            Question.session_id == session_id
        ).order_by(Question.question_number).all()
        return JSONResponse({
            "status": "success",
            "questions": [
                {
                    "number": q.question_number,
                    "question": q.question_text,
                    "answer": q.golden_answer or "",
                }
                for q in questions
            ]
        })
    finally:
        db.close()


@app.post("/api/questions/{session_id}")
async def save_questions(session_id: int, request: Request):
    db = SessionLocal()
    try:
        body = await request.json()
        questions = body.get("questions", [])

        db.query(Question).filter(Question.session_id == session_id).delete()

        for q in questions:
            db.add(Question(
                session_id=session_id,
                question_number=q["number"],
                question_text=q["question"],
                golden_answer=q.get("answer", ""),
            ))

        db.commit()
        return JSONResponse({"status": "success", "count": len(questions)})

    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== RAG ====================

@app.post("/api/rag/{session_id}")
async def run_rag(session_id: int, request: Request):
    """Run RAG pipeline for all questions with all 4 embedding models."""
    db = SessionLocal()
    try:
        body = await request.json()
        apply_llm_overrides(body.get("llm_params", {}))
        chunk_type = body.get("chunk_type", "recursive")
        top_k = body.get("top_k", None)

        # Load embeddings for all models from unified table
        if chunk_type == "all":
            emb_rows = db.query(ChunkEmbedding).filter(
                ChunkEmbedding.session_id == session_id
            ).all()
        else:
            emb_rows = db.query(ChunkEmbedding).filter(
                ChunkEmbedding.session_id == session_id,
                ChunkEmbedding.chunk_type == chunk_type,
            ).all()

        embeddings_by_model = {}
        for key, _, label in EMBEDDING_MODELS:
            col_name = _MODEL_EMB_COL[key]
            embeddings_by_model[key] = [
                {
                    "chunk_text": row.chunk_text,
                    "chunk_type": row.chunk_type,
                    "embedding": emb_from_db(getattr(row, col_name)),
                }
                for row in emb_rows
                if getattr(row, col_name) is not None
            ]
            print(f"  {label}: {len(embeddings_by_model[key])} embeddings loaded")

        if all(len(v) == 0 for v in embeddings_by_model.values()):
            return JSONResponse({"status": "error", "message": "No embeddings found. Run embedding step first."}, status_code=404)

        # Load questions
        questions = db.query(Question).filter(
            Question.session_id == session_id
        ).order_by(Question.question_number).all()

        if not questions:
            return JSONResponse({"status": "error", "message": "No questions found. Save questions first."}, status_code=404)

        q_data = [
            {"question_number": q.question_number, "question_text": q.question_text, "golden_answer": q.golden_answer}
            for q in questions
        ]

        print(f"\n🚀 Running RAG pipeline for session {session_id} ({chunk_type} chunks)...")
        rag_results = run_rag_pipeline(q_data, embeddings_by_model, top_k=top_k)

        # Clear existing EvaluationResult for this session+chunk_type
        print(f"\n🗑️  Clearing existing RAG results for session {session_id} (chunk_type={chunk_type})...")
        db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id,
            EvaluationResult.chunk_type == chunk_type,
        ).delete()
        db.commit()

        # Store results — one row per (question, model)
        for r in rag_results:
            for key, _, _ in EMBEDDING_MODELS:
                mr = r["results_by_model"].get(key)
                if not mr:
                    continue
                chunks_json = json.dumps(
                    [{"text": c["chunk_text"], "sim": round(c.get("similarity", 0), 4), "type": c.get("chunk_type", "")}
                     for c in mr["retrieved_chunks"]],
                    ensure_ascii=False,
                )
                db.add(EvaluationResult(
                    session_id=session_id,
                    question_id=0,
                    question_number=r["question_number"],
                    model_name=key,
                    chunk_type=chunk_type,
                    retrieved_chunks=chunks_json,
                    llm_answer=mr["llm_answer"],
                    llm_prompt=mr.get("llm_prompt", ""),
                    golden_answer=r["golden_answer"],
                ))

        session = db.query(UploadSession).filter(UploadSession.id == session_id).first()
        if session:
            session.status = "rag_done"
        db.commit()

        # Format results for frontend
        formatted = []
        for r in rag_results:
            answers = {}
            chunks  = {}
            for key, _, _ in EMBEDDING_MODELS:
                mr = r["results_by_model"].get(key, {})
                answers[key] = mr.get("llm_answer", "")
                chunks[key]  = [
                    {"text": c["chunk_text"], "similarity": round(c.get("similarity", 0), 4), "chunk_type": c.get("chunk_type", "")}
                    for c in mr.get("retrieved_chunks", [])
                ]
            formatted.append({
                "question_number": r["question_number"],
                "question_text":   r["question_text"],
                "golden_answer":   r["golden_answer"],
                "answers":         answers,
                "chunks":          chunks,
            })

        return JSONResponse({"status": "success", "results": formatted})

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== Evaluation ====================

@app.post("/api/evaluate/{session_id}")
async def evaluate(session_id: int, request: Request):
    """Evaluate RAG answers against golden answers using LLM (all 4 models)."""
    db = SessionLocal()
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        apply_llm_overrides(body.get("llm_params", {}))
        chunk_type = body.get("chunk_type", "recursive")

        # Load results for all models
        results_by_model = {}
        for key, _, _ in EMBEDDING_MODELS:
            rows = db.query(EvaluationResult).filter(
                EvaluationResult.session_id == session_id,
                EvaluationResult.model_name == key,
                EvaluationResult.chunk_type == chunk_type,
            ).order_by(EvaluationResult.question_number).all()
            results_by_model[key] = rows

        if all(len(v) == 0 for v in results_by_model.values()):
            return JSONResponse({"status": "error", "message": "No RAG results found. Run RAG first."}, status_code=404)

        # Collect all question numbers
        all_q_nums = sorted({
            r.question_number
            for rows in results_by_model.values()
            for r in rows
        })

        # Get question texts
        questions = db.query(Question).filter(Question.session_id == session_id).all()
        q_map = {q.question_number: q.question_text for q in questions}

        # Build rag_data for evaluation
        rag_data = []
        for q_num in all_q_nums:
            row_by_model = {}
            for key in results_by_model:
                row_by_model[key] = next((r for r in results_by_model[key] if r.question_number == q_num), None)

            first_row = next((r for r in row_by_model.values() if r is not None), None)
            if not first_row:
                continue

            rag_data.append({
                "question_number":  q_num,
                "question_text":    q_map.get(q_num, ""),
                "golden_answer":    first_row.golden_answer or "",
                "answers_by_model": {k: (r.llm_answer or "") for k, r in row_by_model.items() if r},
            })

        print(f"\n📊 Evaluating RAG results for session {session_id}...")
        evaluations = evaluate_all(rag_data)

        # Build prompt lookup maps from DB rows
        prompt_maps = {key: {r.question_number: r.llm_prompt for r in rows} for key, rows in results_by_model.items()}

        # Update evaluation results in DB
        for ev in evaluations:
            for key, rows in results_by_model.items():
                score = ev.get(f"score_{key}", 0)
                for r in rows:
                    if r.question_number == ev["question_number"]:
                        r.evaluation_text  = ev["evaluation_text"]
                        r.evaluation_score = score

            # Attach prompts and chunks for immediate frontend display
            q_num = ev["question_number"]
            ev["llm_prompts"] = {key: prompt_maps[key].get(q_num, "") for key in results_by_model}
            ev["chunks"] = {
                key: _parse_retrieved_chunks(
                    next((r.retrieved_chunks for r in rows if r.question_number == q_num), None)
                )
                for key, rows in results_by_model.items()
            }

        session = db.query(UploadSession).filter(UploadSession.id == session_id).first()
        if session:
            session.status = "evaluated"
        db.commit()

        return JSONResponse({"status": "success", "evaluations": evaluations})

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== Prompt Management ====================

@app.get("/api/prompt/evaluation")
async def get_eval_prompt():
    return JSONResponse({
        "prompt": get_evaluation_prompt(),
        "is_custom": os.path.exists(PROMPT_FILE),
        "default_prompt": DEFAULT_EVALUATION_PROMPT,
    })


@app.post("/api/prompt/evaluation")
async def save_eval_prompt(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    if body.get("reset"):
        if os.path.exists(PROMPT_FILE):
            os.remove(PROMPT_FILE)
        return JSONResponse({"status": "success", "message": "รีเซ็ต prompt กลับเป็นค่าเริ่มต้นสำเร็จ"})

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"status": "error", "message": "Prompt ว่างเปล่า"}, status_code=400)

    missing = [r for r in REQUIRED_PLACEHOLDERS if r not in prompt]
    if missing:
        return JSONResponse({
            "status": "error",
            "message": f"ขาด placeholder: {', '.join(missing)}",
        }, status_code=400)

    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(prompt)

    return JSONResponse({"status": "success", "message": "บันทึก prompt สำเร็จ"})


# ==================== WER ====================

@app.post("/api/wer/{session_id}")
async def compute_wer_endpoint(session_id: int):
    db = SessionLocal()
    try:
        pages = db.query(OcrPage).filter(
            OcrPage.session_id == session_id
        ).order_by(OcrPage.page_number).all()

        if not pages:
            return JSONResponse({"status": "error", "message": "No OCR pages found"}, status_code=404)

        image_map = {p.page_number: p.image_path for p in pages}
        page_data = [{"page_number": p.page_number, "ocr_text": p.ocr_text} for p in pages]
        wer_results = compute_wer_for_session(page_data)

        db.query(WerResult).filter(WerResult.session_id == session_id).delete()
        for wr in wer_results:
            db.add(WerResult(
                session_id=session_id,
                page_number=wr["page_number"],
                ocr_text_preview=wr["ocr_text"],
                reference_text_preview=wr["reference_text"],
                wer_score=wr["wer_score"],
            ))

            ocr_page = db.query(OcrPage).filter(
                OcrPage.session_id == session_id,
                OcrPage.page_number == wr["page_number"],
            ).first()
            if ocr_page:
                ocr_page.wer_score = wr["wer_score"]

        db.commit()

        avg_wer = sum(w["wer_score"] for w in wer_results if w["wer_score"] >= 0) / max(1, len([w for w in wer_results if w["wer_score"] >= 0]))

        for wr in wer_results:
            wr["image_url"] = _image_url(image_map.get(wr["page_number"], ""))

        return JSONResponse({
            "status": "success",
            "results": wer_results,
            "average_wer": round(avg_wer, 4),
        })

    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== LLM Config ====================

@app.get("/api/llm-config")
async def get_llm_config():
    return JSONResponse({
        "temperature": LLM_TEMPERATURE,
        "top_p":       LLM_TOP_P,
        "max_predict": LLM_MAX_PREDICT,
        "num_ctx":     LLM_NUM_CTX,
    })


# ==================== Status & Results ====================

@app.get("/api/sessions")
async def list_sessions():
    db = SessionLocal()
    try:
        sessions = db.query(UploadSession).order_by(UploadSession.id.desc()).all()
        return JSONResponse({
            "sessions": [
                {
                    "id": s.id,
                    "filename": s.filename,
                    "total_pages": s.total_pages,
                    "status": s.status,
                    "created_at": s.created_at.isoformat() if s.created_at else "",
                }
                for s in sessions
            ]
        })
    finally:
        db.close()


@app.get("/api/results/{session_id}")
async def get_results(session_id: int):
    db = SessionLocal()
    try:
        from collections import defaultdict

        evals = db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id
        ).order_by(EvaluationResult.question_number, EvaluationResult.model_name).all()

        wers = db.query(WerResult).filter(
            WerResult.session_id == session_id
        ).order_by(WerResult.page_number).all()

        ocr_pages = db.query(OcrPage).filter(OcrPage.session_id == session_id).all()
        page_image_map = {p.page_number: p.image_path for p in ocr_pages}

        rec_count = db.query(RecursiveChunk).filter(RecursiveChunk.session_id == session_id).count()
        ag_count  = db.query(AgenticChunk).filter(AgenticChunk.session_id == session_id).count()

        questions = db.query(Question).filter(Question.session_id == session_id).all()
        q_text_map = {q.question_number: q.question_text for q in questions}

        # Pick chunk_type to display
        distinct_cts = list({e.chunk_type for e in evals})
        selected_ct = next(
            (ct for ct in ["all", "recursive", "agentic"] if ct in distinct_cts),
            distinct_cts[0] if distinct_cts else "recursive",
        )

        def parse_chunks(e):
            return _parse_retrieved_chunks(e.retrieved_chunks if e else None)

        # Group evals by (question_number, model_name)
        rag_by_q: dict = defaultdict(dict)
        eval_by_q: dict = defaultdict(dict)
        for e in evals:
            if e.chunk_type == selected_ct:
                rag_by_q[e.question_number][e.model_name]  = e
                eval_by_q[e.question_number][e.model_name] = e

        model_keys = [key for key, _, _ in EMBEDDING_MODELS]

        rag_results = []
        for q_num in sorted(rag_by_q.keys()):
            m = rag_by_q[q_num]
            first_row = next((m[k] for k in model_keys if k in m), None)
            rag_results.append({
                "question_number": q_num,
                "chunk_type":      selected_ct,
                "question_text":   q_text_map.get(q_num, ""),
                "golden_answer":   first_row.golden_answer if first_row else "",
                "answers": {key: (m[key].llm_answer if key in m else "") for key in model_keys},
                "chunks":  {key: parse_chunks(m.get(key))                 for key in model_keys},
            })

        eval_summary = []
        for q_num in sorted(eval_by_q.keys()):
            m = eval_by_q[q_num]
            first_row = next((m[k] for k in model_keys if k in m), None)
            scores = {key: (m[key].evaluation_score if key in m else None) for key in model_keys}
            if any(v is not None for v in scores.values()):
                eval_summary.append({
                    "question_number":  q_num,
                    "question_text":    q_text_map.get(q_num, ""),
                    "golden_answer":    first_row.golden_answer if first_row else "",
                    "answers":          {key: (m[key].llm_answer if key in m else "") for key in model_keys},
                    "scores":           scores,
                    "evaluation_text":  next((m[k].evaluation_text for k in model_keys if k in m and m[k].evaluation_text), ""),
                    "llm_prompts":      {key: (m[key].llm_prompt if key in m else "") for key in model_keys},
                    "chunks":           {key: parse_chunks(m.get(key)) for key in model_keys},
                })

        return JSONResponse({
            "evaluations": [
                {
                    "question_number": e.question_number,
                    "model_name":      e.model_name,
                    "chunk_type":      e.chunk_type,
                    "llm_answer":      e.llm_answer,
                    "golden_answer":   e.golden_answer,
                    "evaluation_text": e.evaluation_text,
                    "evaluation_score": e.evaluation_score,
                }
                for e in evals
            ],
            "wer_results": [
                {
                    "page_number":     w.page_number,
                    "wer_score":       w.wer_score,
                    "ocr_text":        w.ocr_text_preview,
                    "reference_text":  w.reference_text_preview,
                    "image_url":       _image_url(page_image_map.get(w.page_number, "")),
                }
                for w in wers
            ],
            "chunk_counts": {"recursive": rec_count, "agentic": ag_count},
            "rag_results":  rag_results,
            "eval_summary": eval_summary,
        })
    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)

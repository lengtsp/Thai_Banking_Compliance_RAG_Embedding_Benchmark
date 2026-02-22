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
    APP_HOST, APP_PORT, UPLOAD_DIR, BEST_OCR_DIR, EMBEDDING_MODEL_4B, EMBEDDING_MODEL_8B,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_PREDICT, LLM_NUM_CTX, apply_llm_overrides,
)
from database import (
    init_db, SessionLocal, UploadSession, OcrPage,
    RecursiveChunk, AgenticChunk, Embedding4b, Embedding8b,
    Question, EvaluationResult, WerResult,
)
from ocr_service import process_pdf
from chunking_service import create_recursive_chunks, create_agentic_chunks
from embedding_service import get_embeddings_batch, unload_model
from rag_service import run_rag_pipeline
from evaluation_service import evaluate_all, get_evaluation_prompt, DEFAULT_EVALUATION_PROMPT, PROMPT_FILE
from wer_service import compute_wer_for_session

app = FastAPI(title="RAG Embedding Comparison")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads_files")


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
                # Override: clear old data and reuse session
                session = db.query(UploadSession).filter(UploadSession.id == override_session_id).first()
                if session:
                    print(f"\n🔄 Override: clearing old data for session {override_session_id}")
                    db.query(OcrPage).filter(OcrPage.session_id == override_session_id).delete()
                    db.query(RecursiveChunk).filter(RecursiveChunk.session_id == override_session_id).delete()
                    db.query(AgenticChunk).filter(AgenticChunk.session_id == override_session_id).delete()
                    db.query(Embedding4b).filter(Embedding4b.session_id == override_session_id).delete()
                    db.query(Embedding8b).filter(Embedding8b.session_id == override_session_id).delete()
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
                # New session
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

            # Run OCR
            print(f"\n📄 Processing: {file.filename} (Session {session.id})")
            ocr_results = process_pdf(pdf_path, session.id)

            # Store OCR results
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

        # Clear existing chunks for this session before re-chunking
        print(f"\n🗑️  Clearing existing chunks for session {session_id}...")
        db.query(RecursiveChunk).filter(RecursiveChunk.session_id == session_id).delete()
        db.query(AgenticChunk).filter(AgenticChunk.session_id == session_id).delete()
        db.commit()

        # Recursive chunks
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

        # Agentic chunks
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

        # Update session status
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
    """Generate embeddings for all chunks using both 4b and 8b models."""
    db = SessionLocal()
    try:
        # Get all recursive chunks
        rec_chunks = db.query(RecursiveChunk).filter(
            RecursiveChunk.session_id == session_id
        ).order_by(RecursiveChunk.id).all()

        # Get all agentic chunks
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

        # Clear existing embeddings for this session before re-embedding
        print(f"\n🗑️  Clearing existing embeddings for session {session_id}...")
        db.query(Embedding4b).filter(Embedding4b.session_id == session_id).delete()
        db.query(Embedding8b).filter(Embedding8b.session_id == session_id).delete()
        db.commit()

        texts = [c["text"] for c in all_chunks]

        # 4b embeddings — unload after last chunk to free VRAM before loading 8b
        print(f"\n🔵 Generating 4b embeddings for {len(texts)} chunks...")
        embs_4b = get_embeddings_batch(texts, EMBEDDING_MODEL_4B, unload_after=True)
        for i, emb in enumerate(embs_4b):
            db.add(Embedding4b(
                session_id=session_id,
                chunk_id=all_chunks[i]["id"],
                chunk_type=all_chunks[i]["type"],
                chunk_text=all_chunks[i]["text"],
                embedding=emb.tobytes(),
            ))

        # 8b embeddings — unload after last chunk when done
        print(f"\n🟣 Generating 8b embeddings for {len(texts)} chunks...")
        embs_8b = get_embeddings_batch(texts, EMBEDDING_MODEL_8B, unload_after=True)
        for i, emb in enumerate(embs_8b):
            db.add(Embedding8b(
                session_id=session_id,
                chunk_id=all_chunks[i]["id"],
                chunk_type=all_chunks[i]["type"],
                chunk_text=all_chunks[i]["text"],
                embedding=emb.tobytes(),
            ))

        session = db.query(UploadSession).filter(UploadSession.id == session_id).first()
        if session:
            session.status = "embedded"
        db.commit()

        return JSONResponse({
            "status": "success",
            "total_chunks": len(all_chunks),
            "embeddings_4b": len(embs_4b),
            "embeddings_8b": len(embs_8b),
        })

    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        db.close()


# ==================== Questions ====================

@app.get("/api/questions/{session_id}")
async def get_questions(session_id: int):
    """Get saved questions for a session."""
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
    """Save questions and golden answers for a session."""
    db = SessionLocal()
    try:
        body = await request.json()
        questions = body.get("questions", [])

        # Delete existing questions for this session
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
    """Run RAG pipeline for all questions with both embedding models."""
    db = SessionLocal()
    try:
        body = await request.json()
        apply_llm_overrides(body.get("llm_params", {}))
        chunk_type = body.get("chunk_type", "recursive")  # 'recursive', 'agentic', or 'all'
        top_k = body.get("top_k", None)  # None = use default from config

        # Load embeddings — support 'all' by combining both types
        if chunk_type == "all":
            embs_4b_rows = db.query(Embedding4b).filter(
                Embedding4b.session_id == session_id,
            ).all()
            embs_8b_rows = db.query(Embedding8b).filter(
                Embedding8b.session_id == session_id,
            ).all()
        else:
            embs_4b_rows = db.query(Embedding4b).filter(
                Embedding4b.session_id == session_id,
                Embedding4b.chunk_type == chunk_type,
            ).all()
            embs_8b_rows = db.query(Embedding8b).filter(
                Embedding8b.session_id == session_id,
                Embedding8b.chunk_type == chunk_type,
            ).all()

        if not embs_4b_rows or not embs_8b_rows:
            return JSONResponse({"status": "error", "message": "No embeddings found. Run embedding step first."}, status_code=404)

        # Convert stored bytes back to numpy arrays
        embeddings_4b = [
            {"chunk_text": e.chunk_text, "chunk_type": e.chunk_type, "embedding": np.frombuffer(e.embedding, dtype=np.float32).copy()}
            for e in embs_4b_rows
        ]
        embeddings_8b = [
            {"chunk_text": e.chunk_text, "chunk_type": e.chunk_type, "embedding": np.frombuffer(e.embedding, dtype=np.float32).copy()}
            for e in embs_8b_rows
        ]

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

        # Run RAG
        print(f"\n🚀 Running RAG pipeline for session {session_id} ({chunk_type} chunks)...")
        rag_results = run_rag_pipeline(q_data, embeddings_4b, embeddings_8b, top_k=top_k)

        # Clear existing EvaluationResult for this session+chunk_type before storing new ones
        print(f"\n🗑️  Clearing existing RAG results for session {session_id} (chunk_type={chunk_type})...")
        db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id,
            EvaluationResult.chunk_type == chunk_type,
        ).delete()
        db.commit()

        # Store results
        store_chunk_type = chunk_type
        for r in rag_results:
            for model_key, model_name in [("result_4b", "4b"), ("result_8b", "8b")]:
                mr = r[model_key]
                # Store as list of {text, sim, type} to preserve similarity scores and chunk type
                chunks_json = json.dumps(
                    [{"text": c["chunk_text"], "sim": round(c.get("similarity", 0), 4), "type": c.get("chunk_type", "")}
                     for c in mr["retrieved_chunks"]],
                    ensure_ascii=False,
                )
                db.add(EvaluationResult(
                    session_id=session_id,
                    question_id=0,
                    question_number=r["question_number"],
                    model_name=model_name,
                    chunk_type=store_chunk_type,
                    retrieved_chunks=chunks_json,
                    llm_answer=mr["llm_answer"],
                    llm_prompt=mr.get("llm_prompt", ""),
                    golden_answer=r["golden_answer"],
                ))

        session = db.query(UploadSession).filter(UploadSession.id == session_id).first()
        if session:
            session.status = "rag_done"
        db.commit()

        # Format results for frontend (include similarity scores)
        formatted = []
        for r in rag_results:
            formatted.append({
                "question_number": r["question_number"],
                "question_text": r["question_text"],
                "golden_answer": r["golden_answer"],
                "answer_4b": r["result_4b"]["llm_answer"],
                "answer_8b": r["result_8b"]["llm_answer"],
                "chunks_4b": [
                    {"text": c["chunk_text"], "similarity": round(c.get("similarity", 0), 4), "chunk_type": c.get("chunk_type", "")}
                    for c in r["result_4b"]["retrieved_chunks"]
                ],
                "chunks_8b": [
                    {"text": c["chunk_text"], "similarity": round(c.get("similarity", 0), 4), "chunk_type": c.get("chunk_type", "")}
                    for c in r["result_8b"]["retrieved_chunks"]
                ],
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
    """Evaluate RAG answers against golden answers using LLM."""
    db = SessionLocal()
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        apply_llm_overrides(body.get("llm_params", {}))
        chunk_type = body.get("chunk_type", "recursive")

        # Get RAG results
        results_4b = db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id,
            EvaluationResult.model_name == "4b",
            EvaluationResult.chunk_type == chunk_type,
        ).order_by(EvaluationResult.question_number).all()

        results_8b = db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id,
            EvaluationResult.model_name == "8b",
            EvaluationResult.chunk_type == chunk_type,
        ).order_by(EvaluationResult.question_number).all()

        if not results_4b or not results_8b:
            return JSONResponse({"status": "error", "message": "No RAG results found. Run RAG first."}, status_code=404)

        # Build data for evaluation
        rag_data = []
        for r4, r8 in zip(results_4b, results_8b):
            rag_data.append({
                "question_number": r4.question_number,
                "question_text": "",
                "golden_answer": r4.golden_answer or "",
                "result_4b": {"llm_answer": r4.llm_answer or ""},
                "result_8b": {"llm_answer": r8.llm_answer or ""},
            })

        # Get question text
        questions = db.query(Question).filter(
            Question.session_id == session_id
        ).order_by(Question.question_number).all()
        q_map = {q.question_number: q.question_text for q in questions}
        for r in rag_data:
            r["question_text"] = q_map.get(r["question_number"], "")

        # Run evaluation
        print(f"\n📊 Evaluating RAG results for session {session_id}...")
        evaluations = evaluate_all(rag_data)

        # Build prompt lookup maps from DB rows
        prompt_map_4b = {r.question_number: r.llm_prompt for r in results_4b}
        prompt_map_8b = {r.question_number: r.llm_prompt for r in results_8b}

        # Update evaluation results in DB
        for ev in evaluations:
            for r4 in results_4b:
                if r4.question_number == ev["question_number"]:
                    r4.evaluation_text = ev["evaluation_text"]
                    r4.evaluation_score = ev.get("score_4b", 0)
            for r8 in results_8b:
                if r8.question_number == ev["question_number"]:
                    r8.evaluation_text = ev["evaluation_text"]
                    r8.evaluation_score = ev.get("score_8b", 0)
            # Attach prompts so frontend can display them immediately
            ev["llm_prompt_4b"] = prompt_map_4b.get(ev["question_number"], "")
            ev["llm_prompt_8b"] = prompt_map_8b.get(ev["question_number"], "")

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
    """Return current evaluation prompt template (custom or default)."""
    return JSONResponse({
        "prompt": get_evaluation_prompt(),
        "is_custom": os.path.exists(PROMPT_FILE),
        "default_prompt": DEFAULT_EVALUATION_PROMPT,
    })


@app.post("/api/prompt/evaluation")
async def save_eval_prompt(request: Request):
    """Save or reset the evaluation prompt template."""
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

    # Validate required placeholders
    required = ["{question}", "{golden_answer}", "{answer_4b}", "{answer_8b}"]
    missing = [r for r in required if r not in prompt]
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
    """Compute WER for OCR pages against ground truth."""
    db = SessionLocal()
    try:
        pages = db.query(OcrPage).filter(
            OcrPage.session_id == session_id
        ).order_by(OcrPage.page_number).all()

        if not pages:
            return JSONResponse({"status": "error", "message": "No OCR pages found"}, status_code=404)

        # Build image path map from OcrPage
        image_map = {p.page_number: p.image_path for p in pages}
        page_data = [{"page_number": p.page_number, "ocr_text": p.ocr_text} for p in pages]
        wer_results = compute_wer_for_session(page_data)

        # Store WER results (store full text in preview columns — they are Text type)
        db.query(WerResult).filter(WerResult.session_id == session_id).delete()
        for wr in wer_results:
            db.add(WerResult(
                session_id=session_id,
                page_number=wr["page_number"],
                ocr_text_preview=wr["ocr_text"],
                reference_text_preview=wr["reference_text"],
                wer_score=wr["wer_score"],
            ))

            # Also update the OcrPage wer_score
            ocr_page = db.query(OcrPage).filter(
                OcrPage.session_id == session_id,
                OcrPage.page_number == wr["page_number"],
            ).first()
            if ocr_page:
                ocr_page.wer_score = wr["wer_score"]

        db.commit()

        avg_wer = sum(w["wer_score"] for w in wer_results if w["wer_score"] >= 0) / max(1, len([w for w in wer_results if w["wer_score"] >= 0]))

        # Add image_url to each result before returning
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
    """Return current LLM parameter defaults (from .env)."""
    return JSONResponse({
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_predict": LLM_MAX_PREDICT,
        "num_ctx": LLM_NUM_CTX,
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

        # Get all evaluation results
        evals = db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id
        ).order_by(EvaluationResult.question_number, EvaluationResult.model_name).all()

        # Get WER results
        wers = db.query(WerResult).filter(
            WerResult.session_id == session_id
        ).order_by(WerResult.page_number).all()

        # Get OcrPage image paths for WER display
        ocr_pages = db.query(OcrPage).filter(OcrPage.session_id == session_id).all()
        page_image_map = {p.page_number: p.image_path for p in ocr_pages}

        # Get chunks counts
        rec_count = db.query(RecursiveChunk).filter(RecursiveChunk.session_id == session_id).count()
        ag_count = db.query(AgenticChunk).filter(AgenticChunk.session_id == session_id).count()

        # Get question texts
        questions = db.query(Question).filter(Question.session_id == session_id).all()
        q_text_map = {q.question_number: q.question_text for q in questions}

        # Pick chunk_type to display (priority: all > recursive > agentic)
        distinct_cts = list({e.chunk_type for e in evals})
        selected_ct = next(
            (ct for ct in ["all", "recursive", "agentic"] if ct in distinct_cts),
            distinct_cts[0] if distinct_cts else "recursive",
        )

        # Build rag_results grouped by question for selected chunk_type
        rag_by_q: dict = defaultdict(dict)
        for e in evals:
            if e.chunk_type == selected_ct:
                rag_by_q[e.question_number][e.model_name] = e

        def parse_chunks(e):
            if not e or not e.retrieved_chunks:
                return []
            try:
                raw = json.loads(e.retrieved_chunks)
                result = []
                for item in raw:
                    if isinstance(item, dict):
                        # New format: {text, sim, type}
                        result.append({
                            "text": item.get("text", ""),
                            "similarity": item.get("sim", 0),
                            "chunk_type": item.get("type", ""),
                        })
                    else:
                        # Old format: plain string
                        result.append({"text": str(item), "similarity": 0, "chunk_type": ""})
                return result
            except Exception:
                return []

        rag_results = []
        for q_num in sorted(rag_by_q.keys()):
            m = rag_by_q[q_num]
            r4, r8 = m.get("4b"), m.get("8b")
            rag_results.append({
                "question_number": q_num,
                "chunk_type": selected_ct,
                "question_text": q_text_map.get(q_num, ""),
                "golden_answer": (r4 or r8).golden_answer if (r4 or r8) else "",
                "answer_4b": r4.llm_answer if r4 else "",
                "answer_8b": r8.llm_answer if r8 else "",
                "chunks_4b": parse_chunks(r4),
                "chunks_8b": parse_chunks(r8),
            })

        # Build eval_summary (only questions that have been scored)
        eval_by_q: dict = defaultdict(dict)
        for e in evals:
            if e.chunk_type == selected_ct:
                eval_by_q[e.question_number][e.model_name] = e

        eval_summary = []
        for q_num in sorted(eval_by_q.keys()):
            m = eval_by_q[q_num]
            r4, r8 = m.get("4b"), m.get("8b")
            if (r4 and r4.evaluation_score is not None) or (r8 and r8.evaluation_score is not None):
                eval_summary.append({
                    "question_number": q_num,
                    "question_text": q_text_map.get(q_num, ""),
                    "golden_answer": (r4 or r8).golden_answer if (r4 or r8) else "",
                    "answer_4b": r4.llm_answer if r4 else "",
                    "answer_8b": r8.llm_answer if r8 else "",
                    "score_4b": r4.evaluation_score if r4 else None,
                    "score_8b": r8.evaluation_score if r8 else None,
                    "evaluation_text": (r4.evaluation_text if (r4 and r4.evaluation_text) else (r8.evaluation_text if r8 else "")),
                    "llm_prompt_4b": r4.llm_prompt if r4 else "",
                    "llm_prompt_8b": r8.llm_prompt if r8 else "",
                })

        return JSONResponse({
            "evaluations": [
                {
                    "question_number": e.question_number,
                    "model_name": e.model_name,
                    "chunk_type": e.chunk_type,
                    "llm_answer": e.llm_answer,
                    "golden_answer": e.golden_answer,
                    "evaluation_text": e.evaluation_text,
                    "evaluation_score": e.evaluation_score,
                }
                for e in evals
            ],
            "wer_results": [
                {
                    "page_number": w.page_number,
                    "wer_score": w.wer_score,
                    "ocr_text": w.ocr_text_preview,
                    "reference_text": w.reference_text_preview,
                    "image_url": _image_url(page_image_map.get(w.page_number, "")),
                }
                for w in wers
            ],
            "chunk_counts": {"recursive": rec_count, "agentic": ag_count},
            "rag_results": rag_results,
            "eval_summary": eval_summary,
        })
    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)

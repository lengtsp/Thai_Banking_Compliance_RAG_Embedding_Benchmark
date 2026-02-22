import numpy as np
import httpx
from config import (
    OLLAMA_BASE_URL, LLM_MODEL, RAG_TOP_K,
    EMBEDDING_MODELS,
    llm_options,
)
from embedding_service import get_embedding, retrieve_top_k

# Quick lookup: model key → ollama model name
_MODEL_OLLAMA = {key: model for key, model, _ in EMBEDDING_MODELS}
_MODEL_LABEL  = {key: label for key, _, label in EMBEDDING_MODELS}


def generate_answer(question: str, context_chunks: list[dict]) -> dict:
    """Send question + retrieved context to LLM for answer generation.
    Returns {"answer": str, "prompt": str}.
    """
    context_text = "\n\n---\n\n".join([
        f"[Chunk {i+1}] (similarity: {c.get('similarity', 0):.4f}):\n{c['chunk_text']}"
        for i, c in enumerate(context_chunks)
    ])

    prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านกฎระเบียบการเงินของธนาคารแห่งประเทศไทย

จากเนื้อหาอ้างอิงต่อไปนี้:

{context_text}

---

คำถาม: {question}

กรุณาตอบคำถามข้างต้นอย่างกระชับ ตรงประเด็น โดยอ้างอิงจากเนื้อหาที่ให้มา ตอบเป็นภาษาไทย ไม่ต้องอธิบายยืดยาว ให้ตอบสั้นๆ เฉพาะใจความสำคัญ"""

    print(f"\n{'─'*60}")
    print(f"📝 [RAG Prompt] → {LLM_MODEL}")
    print(prompt)
    print(f"{'─'*60}\n")

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": llm_options(),
    }

    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()
    print(f"💬 [RAG Answer preview]: {answer[:200]}\n")
    return {"answer": answer, "prompt": prompt}


def run_rag_pipeline(
    questions: list[dict],
    embeddings_by_model: dict,   # {model_key: [{"chunk_text": str, "embedding": np.ndarray, ...}]}
    top_k: int = None,
) -> list[dict]:
    """
    Run full RAG pipeline for all questions across all embedding models.
    Per question: for each model → retrieve top-k → generate answer.

    Returns list of:
    {
        "question_number": int,
        "question_text": str,
        "golden_answer": str,
        "results_by_model": {
            model_key: {
                "retrieved_chunks": [...],
                "llm_answer": str,
                "llm_prompt": str,
            }
        }
    }
    """
    if top_k is None:
        top_k = RAG_TOP_K

    model_keys = [key for key, _, _ in EMBEDDING_MODELS if key in embeddings_by_model]
    all_results = []

    for q in questions:
        q_num  = q["question_number"]
        q_text = q["question_text"]
        print(f"\n{'='*60}")
        print(f"  📝 ข้อที่ {q_num}: {q_text[:80]}...")
        print(f"{'='*60}")

        results_by_model = {}
        for step, key in enumerate(model_keys, 1):
            label       = _MODEL_LABEL.get(key, key)
            ollama_model = _MODEL_OLLAMA.get(key, key)
            embeddings   = embeddings_by_model[key]

            print(f"\n  [Step {step}A] {label} — Retrieving top-{top_k} chunks...")
            query_emb  = get_embedding(q_text, ollama_model)
            top_chunks = retrieve_top_k(query_emb, embeddings, top_k)
            top_sim    = top_chunks[0]["similarity"] if top_chunks else 0
            print(f"           Retrieved {len(top_chunks)} chunks | top similarity: {top_sim:.4f}")

            print(f"\n  [Step {step}B] {label} — Sending to LLM for answer...")
            gen = generate_answer(q_text, top_chunks)
            print(f"           Answer (preview): {gen['answer'][:120]}...")

            results_by_model[key] = {
                "retrieved_chunks": top_chunks,
                "llm_answer":       gen["answer"],
                "llm_prompt":       gen["prompt"],
            }

        print(f"\n  ✅ ข้อที่ {q_num} เสร็จสมบูรณ์")

        all_results.append({
            "question_number":  q_num,
            "question_text":    q_text,
            "golden_answer":    q.get("golden_answer", ""),
            "results_by_model": results_by_model,
        })

    return all_results

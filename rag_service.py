import numpy as np
import httpx
from config import (
    OLLAMA_BASE_URL, LLM_MODEL, RAG_TOP_K,
    EMBEDDING_MODEL_4B, EMBEDDING_MODEL_8B,
    llm_options,
)
from embedding_service import get_embedding, retrieve_top_k


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
    embeddings_4b: list[dict],
    embeddings_8b: list[dict],
    top_k: int = None,
) -> list[dict]:
    """
    Run full RAG pipeline for all questions, one question at a time.
    Per question order: 4B retrieve → 4B LLM → 8B retrieve → 8B LLM.
    Returns list of results per question.
    """
    if top_k is None:
        top_k = RAG_TOP_K

    all_results = []

    for q in questions:
        q_num = q["question_number"]
        q_text = q["question_text"]
        print(f"\n{'='*60}")
        print(f"  📝 ข้อที่ {q_num}: {q_text[:80]}...")
        print(f"{'='*60}")

        # --- Step 1: 4B Retrieve ---
        print(f"\n  [Step 1] 🔵 4B Embedding — Retrieving top-{top_k} chunks...")
        query_emb_4b = get_embedding(q_text, EMBEDDING_MODEL_4B)
        top_chunks_4b = retrieve_top_k(query_emb_4b, embeddings_4b, top_k)
        top_sim_4b = top_chunks_4b[0]["similarity"] if top_chunks_4b else 0
        print(f"           Retrieved {len(top_chunks_4b)} chunks | top similarity: {top_sim_4b:.4f}")

        # --- Step 2: 4B LLM Answer ---
        print(f"\n  [Step 2] 🔵 4B — Sending to LLM for answer...")
        result_4b_gen = generate_answer(q_text, top_chunks_4b)
        answer_4b, prompt_4b = result_4b_gen["answer"], result_4b_gen["prompt"]
        print(f"           Answer (preview): {answer_4b[:120]}...")

        # --- Step 3: 8B Retrieve ---
        print(f"\n  [Step 3] 🟣 8B Embedding — Retrieving top-{top_k} chunks...")
        query_emb_8b = get_embedding(q_text, EMBEDDING_MODEL_8B)
        top_chunks_8b = retrieve_top_k(query_emb_8b, embeddings_8b, top_k)
        top_sim_8b = top_chunks_8b[0]["similarity"] if top_chunks_8b else 0
        print(f"           Retrieved {len(top_chunks_8b)} chunks | top similarity: {top_sim_8b:.4f}")

        # --- Step 4: 8B LLM Answer ---
        print(f"\n  [Step 4] 🟣 8B — Sending to LLM for answer...")
        result_8b_gen = generate_answer(q_text, top_chunks_8b)
        answer_8b, prompt_8b = result_8b_gen["answer"], result_8b_gen["prompt"]
        print(f"           Answer (preview): {answer_8b[:120]}...")

        print(f"\n  ✅ ข้อที่ {q_num} เสร็จสมบูรณ์")

        all_results.append({
            "question_number": q_num,
            "question_text": q_text,
            "golden_answer": q.get("golden_answer", ""),
            "result_4b": {
                "model_name": "4b",
                "retrieved_chunks": top_chunks_4b,
                "llm_answer": answer_4b,
                "llm_prompt": prompt_4b,
            },
            "result_8b": {
                "model_name": "8b",
                "retrieved_chunks": top_chunks_8b,
                "llm_answer": answer_8b,
                "llm_prompt": prompt_8b,
            },
        })

    return all_results

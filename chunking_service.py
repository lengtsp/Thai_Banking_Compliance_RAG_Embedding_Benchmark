import json
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import OLLAMA_BASE_URL, LLM_MODEL, RECURSIVE_CHUNK_SIZE, RECURSIVE_CHUNK_OVERLAP, llm_options


def create_recursive_chunks(pages: list[dict]) -> list[dict]:
    """
    Create recursive character text chunks from OCR pages.
    Each page is chunked independently to preserve page-level boundaries.
    Returns list of {page_number, chunk_index, chunk_text, chunk_size}.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    all_chunks = []
    for page in pages:
        page_num = page["page_number"]
        text = page["ocr_text"] or ""
        if not text.strip():
            continue

        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "page_number": page_num,
                "chunk_index": idx,
                "chunk_text": chunk,
                "chunk_size": len(chunk),
            })

    return all_chunks


def create_agentic_chunks(pages: list[dict]) -> list[dict]:
    """
    Use LLM to create semantic/agentic chunks from OCR pages.
    Each page is sent to the LLM to break into meaningful semantic sections.
    Returns list of {page_number, chunk_index, chunk_text, chunk_title, chunk_size}.
    """
    all_chunks = []

    for page in pages:
        page_num = page["page_number"]
        text = page["ocr_text"] or ""
        if not text.strip():
            continue

        prompt = f"""คุณเป็นผู้เชี่ยวชาญในการแบ่งเอกสารออกเป็นส่วนย่อยเชิงความหมาย (semantic chunking)

จากข้อความต่อไปนี้ ซึ่งเป็นเนื้อหาจากหน้าที่ {page_num} ของเอกสาร:

---
{text}
---

กรุณาแบ่งข้อความนี้ออกเป็น chunks เชิงความหมาย โดยแต่ละ chunk ควร:
1. มีเนื้อหาที่ครบถ้วนในตัวเอง (self-contained)
2. มีหัวข้อหรือประเด็นหลักเดียว
3. มีความยาวไม่เกิน 1500 ตัวอักษร

ตอบเป็น JSON array เท่านั้น โดยแต่ละ element มี:
- "title": หัวข้อสั้นๆ ของ chunk (ภาษาไทย)
- "text": เนื้อหาของ chunk

ตัวอย่างรูปแบบ:
[
  {{"title": "หัวข้อ 1", "text": "เนื้อหา..."}},
  {{"title": "หัวข้อ 2", "text": "เนื้อหา..."}}
]

ตอบเป็น JSON array เท่านั้น ห้ามมี text อื่นนอกเหนือจาก JSON"""

        try:
            print(f"\n{'─'*60}")
            print(f"📝 [Agentic Chunk Prompt — Page {page_num}] → {LLM_MODEL}")
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
                llm_response = result.get("response", "").strip()

            # Parse JSON from LLM response
            # Try to extract JSON array from the response
            json_start = llm_response.find("[")
            json_end = llm_response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                chunks_data = json.loads(json_str)
            else:
                # Fallback: treat entire page as one chunk
                chunks_data = [{"title": f"หน้า {page_num}", "text": text}]

            for idx, chunk_data in enumerate(chunks_data):
                chunk_text = chunk_data.get("text", "")
                all_chunks.append({
                    "page_number": page_num,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "chunk_title": chunk_data.get("title", ""),
                    "chunk_size": len(chunk_text),
                })

            print(f"  🧠 Agentic chunking page {page_num}: {len(chunks_data)} chunks")

        except Exception as e:
            print(f"  ⚠️ Agentic chunking failed for page {page_num}: {e}")
            # Fallback: treat entire page as one chunk
            all_chunks.append({
                "page_number": page_num,
                "chunk_index": 0,
                "chunk_text": text,
                "chunk_title": f"หน้า {page_num} (fallback)",
                "chunk_size": len(text),
            })

    return all_chunks

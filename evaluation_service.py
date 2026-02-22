import os
import re
import httpx
from config import OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODELS, llm_options

PROMPT_FILE = "evaluation_prompt.txt"

DEFAULT_EVALUATION_PROMPT = """คุณเป็นผู้ประเมินคุณภาพคำตอบจากระบบ RAG (Retrieval-Augmented Generation)

**คำถาม:**
{question}

**เฉลย (Golden Answer) — ใช้เป็นเกณฑ์อ้างอิงใจความสำคัญ:**
{golden_answer}

**คำตอบจากโมเดล Embedding 0.6B:**
{answer_06b}

**คำตอบจากโมเดล Embedding 4B:**
{answer_4b}

**คำตอบจากโมเดล Embedding 8B:**
{answer_8b}

**คำตอบจากโมเดล Embedding BGE-M3:**
{answer_bgem3}

---

## หลักการให้คะแนน (สำคัญมาก — อ่านให้ครบก่อนให้คะแนน)

ให้คะแนนโดยเน้นที่ **ใจความสำคัญ** ไม่ใช่ความเหมือนคำต่อคำกับเฉลย เพราะคำตอบที่ดีอาจใช้สำนวนต่างกันแต่มีความหมายเดียวกัน

- **คะแนนสูง (70–100)**: คำตอบครอบคลุมใจความสำคัญของเฉลยครบถ้วน แม้จะใช้ถ้อยคำต่างกัน อนุโลมการเรียบเรียงใหม่ ตราบที่ข้อมูลหลักถูกต้องและครบ
- **คะแนนกลาง (40–69)**: คำตอบมีใจความถูกต้องบางส่วน หรือขาดรายละเอียดสำคัญบางจุด
- **คะแนนต่ำ (0–39)**: คำตอบผิดพลาดในประเด็นหลัก หรือขาดใจความสำคัญหลักไป

ประเด็นที่ใช้ประเมิน:
1. **ใจความครบถ้วน**: ข้อมูลหลักในเฉลยปรากฏอยู่ในคำตอบหรือไม่ (ไม่ต้องคำต่อคำ)
2. **ความถูกต้อง**: ข้อมูลที่ให้มาถูกต้องตามข้อเท็จจริงหรือไม่
3. **ความเกี่ยวข้อง**: คำตอบตอบตรงคำถามหรือไม่

---

## รูปแบบการตอบ (ต้องทำตามอย่างเคร่งครัด)

ตอบในโครงสร้างต่อไปนี้เท่านั้น ห้ามเพิ่มหรือเปลี่ยนแปลงโครงสร้าง:

## การวิเคราะห์

### โมเดล 0.6B
- ใจความสำคัญที่ครบ: [ระบุจุดที่ตรงกับเฉลย]
- ใจความที่ขาด/ผิด: [ระบุถ้ามี ถ้าไม่มีให้ระบุ "ไม่มี"]

### โมเดล 4B
- ใจความสำคัญที่ครบ: [ระบุจุดที่ตรงกับเฉลย]
- ใจความที่ขาด/ผิด: [ระบุถ้ามี ถ้าไม่มีให้ระบุ "ไม่มี"]

### โมเดล 8B
- ใจความสำคัญที่ครบ: [ระบุจุดที่ตรงกับเฉลย]
- ใจความที่ขาด/ผิด: [ระบุถ้ามี ถ้าไม่มีให้ระบุ "ไม่มี"]

### โมเดล BGE-M3
- ใจความสำคัญที่ครบ: [ระบุจุดที่ตรงกับเฉลย]
- ใจความที่ขาด/ผิด: [ระบุถ้ามี ถ้าไม่มีให้ระบุ "ไม่มี"]

### สรุป
[สรุปการเปรียบเทียบ ว่าโมเดลใดให้คำตอบดีกว่าและเพราะเหตุใด]

---SCORES---
SCORE_06B: [ตัวเลขเท่านั้น เช่น 75 ห้ามใส่ ** หรือ text อื่นในบรรทัดนี้]
SCORE_4B: [ตัวเลขเท่านั้น เช่น 80 ห้ามใส่ ** หรือ text อื่นในบรรทัดนี้]
SCORE_8B: [ตัวเลขเท่านั้น เช่น 90 ห้ามใส่ ** หรือ text อื่นในบรรทัดนี้]
SCORE_BGEM3: [ตัวเลขเท่านั้น เช่น 85 ห้ามใส่ ** หรือ text อื่นในบรรทัดนี้]

หมายเหตุ: บรรทัด SCORE_* ต้องอยู่หลัง ---SCORES--- เท่านั้น ห้ามระบุคะแนนในส่วนการวิเคราะห์"""

# Required placeholders for prompt validation
REQUIRED_PLACEHOLDERS = ["{question}", "{golden_answer}", "{answer_06b}", "{answer_4b}", "{answer_8b}", "{answer_bgem3}"]

# Score labels to extract per model key
_SCORE_LABELS = {key: f"SCORE_{key.upper()}" for key, _, _ in EMBEDDING_MODELS}


def get_evaluation_prompt() -> str:
    """Load evaluation prompt template from file if exists, otherwise return default."""
    try:
        if os.path.exists(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return DEFAULT_EVALUATION_PROMPT


def evaluate_answer(
    question: str,
    golden_answer: str,
    answers_by_model: dict,   # {model_key: answer_text}
) -> dict:
    """
    Send all model answers + golden answer to LLM for comparative evaluation.
    Returns {evaluation_text, score_06b, score_4b, score_8b, score_bgem3}.
    """
    prompt_template = get_evaluation_prompt()

    # Build format kwargs — fall back to "(ไม่มีคำตอบ)" for missing models
    fmt_kwargs = {
        "question":      question,
        "golden_answer": golden_answer,
        "answer_06b":    answers_by_model.get("06b",   "(ไม่มีคำตอบ)"),
        "answer_4b":     answers_by_model.get("4b",    "(ไม่มีคำตอบ)"),
        "answer_8b":     answers_by_model.get("8b",    "(ไม่มีคำตอบ)"),
        "answer_bgem3":  answers_by_model.get("bgem3", "(ไม่มีคำตอบ)"),
    }

    try:
        prompt = prompt_template.format(**fmt_kwargs)
    except KeyError:
        # Custom prompt may have fewer placeholders — format what we can
        from string import Formatter
        used_keys = {fname for _, fname, _, _ in Formatter().parse(prompt_template) if fname}
        prompt = prompt_template.format(**{k: v for k, v in fmt_kwargs.items() if k in used_keys})

    print(f"\n{'─'*60}")
    print(f"📝 [Evaluation Prompt] → {LLM_MODEL}")
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
        evaluation_text = result.get("response", "").strip()
    print(f"💬 [Eval response preview]: {evaluation_text[:300]}\n")

    # Parse scores for all models
    scores = {key: _extract_score(evaluation_text, label) for key, label in _SCORE_LABELS.items()}

    return {"evaluation_text": evaluation_text, **{f"score_{k}": v for k, v in scores.items()}}


def _extract_score(text: str, label: str) -> float:
    """Extract numeric score from evaluation text."""
    try:
        for line in text.split("\n"):
            if label in line:
                clean_line = line.replace("**", "").replace("*", "")
                parts = clean_line.split(label)
                if len(parts) > 1:
                    num_str = parts[1].strip().strip(":").strip()
                    match = re.search(r'\d+(?:\.\d+)?', num_str)
                    if match:
                        return float(match.group())
    except Exception:
        pass
    return 0.0


def evaluate_all(rag_results: list[dict]) -> list[dict]:
    """
    Evaluate all RAG results against golden answers.

    Each item in rag_results must have:
        question_number, question_text, golden_answer, answers_by_model
    """
    evaluations = []
    for r in rag_results:
        q_num = r["question_number"]
        print(f"  📊 Evaluating question {q_num}...")

        eval_result = evaluate_answer(
            question=r["question_text"],
            golden_answer=r["golden_answer"],
            answers_by_model=r["answers_by_model"],
        )

        evaluations.append({
            "question_number":  q_num,
            "question_text":    r["question_text"],
            "golden_answer":    r["golden_answer"],
            "answers_by_model": r["answers_by_model"],
            **eval_result,
        })

    return evaluations

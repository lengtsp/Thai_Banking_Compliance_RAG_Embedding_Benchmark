import os
import re
from jiwer import wer as compute_wer
from config import BEST_OCR_DIR


def normalize_text(text: str) -> str:
    """Normalize text for WER — keep only Unicode letters and digits.

    Strips markdown, whitespace, newlines, and all special characters so that
    WER reflects only actual content differences, not formatting differences.
    Thai and other Unicode characters are preserved.
    """
    # Extract only Unicode letters and digits (includes Thai, Latin, CJK, numbers)
    # [^\W_]+ matches any word character except underscore → letters + digits
    tokens = re.findall(r'[^\W_]+', text, re.UNICODE)
    return ' '.join(tokens)


def compute_page_wer(ocr_text: str, reference_text: str) -> float:
    """Compute Word Error Rate between OCR output and reference text."""
    norm_ocr = normalize_text(ocr_text)
    norm_ref = normalize_text(reference_text)

    if not norm_ref.strip():
        return 0.0
    if not norm_ocr.strip():
        return 1.0

    try:
        score = compute_wer(norm_ref, norm_ocr)
        return round(score, 4)
    except Exception as e:
        print(f"  ⚠️ WER computation error: {e}")
        return -1.0


def compute_wer_for_session(ocr_pages: list[dict], best_ocr_dir: str = None) -> list[dict]:
    """
    Compute WER for all pages in a session against ground truth.
    ocr_pages: list of {page_number, ocr_text}
    Returns list of {page_number, wer_score, ocr_text, reference_text}.
    """
    if best_ocr_dir is None:
        best_ocr_dir = BEST_OCR_DIR

    results = []
    for page in ocr_pages:
        page_num = page["page_number"]
        ocr_text = page.get("ocr_text", "")

        ref_file = os.path.join(best_ocr_dir, f"page_{page_num}.txt")
        if not os.path.exists(ref_file):
            results.append({
                "page_number": page_num,
                "wer_score": -1.0,
                "ocr_text": ocr_text,
                "reference_text": "(reference file not found)",
            })
            continue

        with open(ref_file, "r", encoding="utf-8") as f:
            reference_text = f.read()

        score = compute_page_wer(ocr_text, reference_text)
        results.append({
            "page_number": page_num,
            "wer_score": score,
            "ocr_text": ocr_text,
            "reference_text": reference_text,
        })

    return results

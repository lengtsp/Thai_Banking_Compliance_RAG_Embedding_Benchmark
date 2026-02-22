import os
import base64
import httpx
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from config import OLLAMA_BASE_URL, OCR_MODEL, UPLOAD_DIR


def pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    """Convert each page of a PDF to a JPG image. Returns list of image paths."""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 300 DPI for good OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)

        img_path = os.path.join(output_dir, f"page_{page_num + 1}.jpg")
        pix.save(img_path)
        image_paths.append(img_path)

    doc.close()
    return image_paths


def ocr_image(image_path: str) -> str:
    """Send an image to Ollama vision model for OCR. Returns extracted text."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    base64_image = base64.b64encode(image_data).decode("utf-8")

    prompt = (
        "Please perform OCR on this document image. "
        "Extract ALL text content exactly as it appears, preserving the original formatting, "
        "structure, and language (Thai/English). "
        "Output ONLY the extracted text, no commentary."
    )

    payload = {
        "model": OCR_MODEL,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 12000,
        }
    }

    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()


def process_pdf(pdf_path: str, session_id: int) -> list[dict]:
    """Full pipeline: PDF -> images -> OCR for each page. Returns list of page results."""
    output_dir = os.path.join(UPLOAD_DIR, f"session_{session_id}", "pages")
    image_paths = pdf_to_images(pdf_path, output_dir)

    results = []
    for i, img_path in enumerate(image_paths):
        page_num = i + 1
        print(f"  🔍 OCR page {page_num}/{len(image_paths)}...")
        ocr_text = ocr_image(img_path)
        results.append({
            "page_number": page_num,
            "image_path": img_path,
            "ocr_text": ocr_text,
        })

    return results

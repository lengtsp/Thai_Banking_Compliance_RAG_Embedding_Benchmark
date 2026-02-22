import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

# ===== Database =====
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "test_embedding")
DB_USER = os.getenv("DB_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")

DB_PASSWORD_ENCODED = urllib.parse.quote_plus(DB_PASSWORD)
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ===== Ollama =====
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:7869")

OCR_MODEL = os.getenv("OCR_MODEL", "qwen3-vl:30b-a3b-instruct-q8_0")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:120b")

EMBEDDING_MODEL_4B = os.getenv("EMBEDDING_MODEL_4B", "qwen3-embedding:4b-q8_0")
EMBEDDING_DIM_4B = int(os.getenv("EMBEDDING_DIM_4B", "2560"))

EMBEDDING_MODEL_8B = os.getenv("EMBEDDING_MODEL_8B", "qwen3-embedding:8b")
EMBEDDING_DIM_8B = int(os.getenv("EMBEDDING_DIM_8B", "4096"))

# ===== LLM Options =====
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE",  "0.6"))
LLM_TOP_P        = float(os.getenv("LLM_TOP_P",        "0.95"))
LLM_MAX_PREDICT  = int(os.getenv("LLM_MAX_PREDICT",    "25000"))
# LLM_NUM_CTX = 0 หมายถึงไม่ส่งค่า (ใช้ default ของ Ollama)
LLM_NUM_CTX      = int(os.getenv("LLM_NUM_CTX",        "50000"))

# Runtime overrides (set from UI — takes effect until server restart)
_runtime_llm_overrides: dict = {}


def apply_llm_overrides(params: dict):
    """Apply runtime LLM parameter overrides from UI. None or missing = keep env default."""
    global _runtime_llm_overrides
    _runtime_llm_overrides = {}
    if params.get("temperature") is not None:
        _runtime_llm_overrides["temperature"] = float(params["temperature"])
    if params.get("top_p") is not None:
        _runtime_llm_overrides["top_p"] = float(params["top_p"])
    if params.get("max_predict") is not None:
        _runtime_llm_overrides["num_predict"] = int(params["max_predict"])
    num_ctx = params.get("num_ctx")
    if num_ctx is not None and int(num_ctx) > 0:
        _runtime_llm_overrides["num_ctx"] = int(num_ctx)
    elif "num_ctx" in _runtime_llm_overrides:
        del _runtime_llm_overrides["num_ctx"]


def llm_options() -> dict:
    """Build Ollama options dict. UI overrides take priority over env config."""
    opts = {
        "temperature": LLM_TEMPERATURE,
        "top_p":       LLM_TOP_P,
        "num_predict": LLM_MAX_PREDICT,
    }
    if LLM_NUM_CTX > 0:
        opts["num_ctx"] = LLM_NUM_CTX
    opts.update(_runtime_llm_overrides)
    return opts


# ===== Chunking =====
RECURSIVE_CHUNK_SIZE = int(os.getenv("RECURSIVE_CHUNK_SIZE", "1300"))
RECURSIVE_CHUNK_OVERLAP = int(os.getenv("RECURSIVE_CHUNK_OVERLAP", "30"))

# ===== RAG =====
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# ===== App =====
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
BEST_OCR_DIR = os.getenv("BEST_OCR_DIR", "best_ocr")

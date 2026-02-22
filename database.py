import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, LargeBinary, text
from sqlalchemy.orm import sessionmaker, declarative_base
from config import DB_URL, EMBEDDING_DIM_06B, EMBEDDING_DIM_4B, EMBEDDING_DIM_8B, EMBEDDING_DIM_BGEM3

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ──────────────────────────────────────────────────────────
# pgvector detection (runs once at import time)
# ──────────────────────────────────────────────────────────
def _detect_pgvector() -> bool:
    """Return True if pgvector Python package AND PostgreSQL extension are both available."""
    try:
        from pgvector.sqlalchemy import Vector  # Python package check
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        print("✅ pgvector extension enabled — using VECTOR columns")
        return True
    except Exception as e:
        print(f"  ℹ️  pgvector unavailable ({type(e).__name__}): using BYTEA fallback")
        return False


USE_PGVECTOR: bool = _detect_pgvector()


def _emb_col(dim: int) -> Column:
    """Return Vector(dim) column if pgvector is available, else LargeBinary (bytea)."""
    if USE_PGVECTOR:
        from pgvector.sqlalchemy import Vector
        return Column(Vector(dim))
    return Column(LargeBinary)


# ──────────────────────────────────────────────────────────
# ORM Models
# ──────────────────────────────────────────────────────────
class UploadSession(Base):
    __tablename__ = "upload_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    total_pages = Column(Integer, default=0)
    status = Column(String(50), default="uploaded")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class OcrPage(Base):
    __tablename__ = "ocr_pages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    image_path = Column(String(1000))
    ocr_text = Column(Text)
    wer_score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class RecursiveChunk(Base):
    __tablename__ = "recursive_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    page_number = Column(Integer)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class AgenticChunk(Base):
    __tablename__ = "agentic_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    page_number = Column(Integer)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_title = Column(String(500))
    chunk_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ChunkEmbedding(Base):
    """Single table holding all 4 embedding model vectors per chunk."""
    __tablename__ = "chunk_embeddings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    chunk_id = Column(Integer, nullable=False)
    chunk_type = Column(String(20), nullable=False)
    chunk_text = Column(Text)
    embedding_06b = _emb_col(EMBEDDING_DIM_06B)
    embedding_4b = _emb_col(EMBEDDING_DIM_4B)
    embedding_8b = _emb_col(EMBEDDING_DIM_8B)
    embedding_bgem3 = _emb_col(EMBEDDING_DIM_BGEM3)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    question_number = Column(Integer, nullable=False)
    question_text = Column(Text, nullable=False)
    golden_answer = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    question_id = Column(Integer, nullable=False)
    question_number = Column(Integer)
    model_name = Column(String(100))          # '06b', '4b', '8b', 'bgem3'
    chunk_type = Column(String(20))
    retrieved_chunks = Column(Text)           # JSON list
    llm_answer = Column(Text)
    llm_prompt = Column(Text)
    golden_answer = Column(Text)
    evaluation_text = Column(Text)
    evaluation_score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class WerResult(Base):
    __tablename__ = "wer_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    ocr_text_preview = Column(Text)
    reference_text_preview = Column(Text)
    wer_score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


_OLD_EMBEDDING_TABLES = ["embeddings_06b", "embeddings_4b", "embeddings_8b", "embeddings_bgem3"]
_EMBEDDING_TABLES = ["chunk_embeddings"]


# ──────────────────────────────────────────────────────────
# Helpers: embedding save / load (handles both column types)
# ──────────────────────────────────────────────────────────
import numpy as np


def emb_to_db(arr: np.ndarray):
    """Convert numpy array to value suitable for the embedding column."""
    return arr.tolist() if USE_PGVECTOR else arr.tobytes()


def emb_from_db(value) -> np.ndarray:
    """Convert stored embedding column value back to numpy float32 array."""
    if USE_PGVECTOR:
        return np.array(value, dtype=np.float32)
    return np.frombuffer(value, dtype=np.float32).copy()


# ──────────────────────────────────────────────────────────
# DB initialisation
# ──────────────────────────────────────────────────────────
def init_db():
    """Create all tables and run lightweight migrations."""
    with engine.connect() as conn:
        # Migrate: drop old separate embedding tables (replaced by chunk_embeddings)
        for table_name in _OLD_EMBEDDING_TABLES:
            try:
                result = conn.execute(text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    f"WHERE table_name = '{table_name}')"
                )).fetchone()
                if result and result[0]:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    conn.commit()
                    print(f"  ♻️  Dropped old table {table_name} (consolidated into chunk_embeddings)")
            except Exception:
                pass

        if USE_PGVECTOR:
            # Migrate: drop chunk_embeddings if it still uses bytea columns
            try:
                result = conn.execute(text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = 'chunk_embeddings' AND column_name = 'embedding_06b'"
                )).fetchone()
                if result and result[0] == "bytea":
                    conn.execute(text("DROP TABLE IF EXISTS chunk_embeddings"))
                    conn.commit()
                    print("  ♻️  Dropped chunk_embeddings (bytea→vector migration; re-run embedding step)")
            except Exception:
                pass

        # Add llm_prompt column if missing (older schema)
        try:
            conn.execute(text("ALTER TABLE evaluation_results ADD COLUMN llm_prompt TEXT"))
            conn.commit()
        except Exception:
            pass

    Base.metadata.create_all(bind=engine)
    col_type = "VECTOR" if USE_PGVECTOR else "BYTEA (pgvector unavailable)"
    print(f"✅ All tables ready  [embedding column: {col_type}]")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()

import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, LargeBinary, text
from sqlalchemy.orm import sessionmaker, declarative_base
from config import DB_URL

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UploadSession(Base):
    __tablename__ = "upload_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    total_pages = Column(Integer, default=0)
    status = Column(String(50), default="uploaded")  # uploaded, ocr_done, chunked, embedded, rag_done, evaluated
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


class Embedding4b(Base):
    __tablename__ = "embeddings_4b"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    chunk_id = Column(Integer, nullable=False)
    chunk_type = Column(String(20), nullable=False)   # 'recursive' or 'agentic'
    chunk_text = Column(Text)
    embedding = Column(LargeBinary)   # numpy float32 bytes
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Embedding8b(Base):
    __tablename__ = "embeddings_8b"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False)
    chunk_id = Column(Integer, nullable=False)
    chunk_type = Column(String(20), nullable=False)
    chunk_text = Column(Text)
    embedding = Column(LargeBinary)   # numpy float32 bytes
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
    model_name = Column(String(100))          # '4b' or '8b'
    chunk_type = Column(String(20))           # 'recursive' or 'agentic'
    retrieved_chunks = Column(Text)           # JSON list of chunk texts
    llm_answer = Column(Text)
    llm_prompt = Column(Text)                 # Full prompt sent to LLM for RAG inference
    golden_answer = Column(Text)
    evaluation_text = Column(Text)            # LLM evaluation output
    evaluation_score = Column(Float)          # 0-100 score
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


def init_db():
    """Create all tables and run lightweight migrations for new columns."""
    Base.metadata.create_all(bind=engine)
    # Migrate: add llm_prompt column to existing evaluation_results tables
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE evaluation_results ADD COLUMN llm_prompt TEXT"))
            conn.commit()
        except Exception:
            pass  # Column already exists — safe to ignore
    print("✅ All tables ready.")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()

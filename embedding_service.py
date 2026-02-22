import numpy as np
import httpx
from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL_4B, EMBEDDING_DIM_4B,
    EMBEDDING_MODEL_8B, EMBEDDING_DIM_8B,
)


def get_embedding(text: str, model: str, keep_alive: int = -1) -> np.ndarray:
    """Get embedding vector from Ollama for a single text."""
    payload = {
        "model": model,
        "input": text,
        "keep_alive": keep_alive,
    }

    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        embeddings = result.get("embeddings", [[]])
        return np.array(embeddings[0], dtype=np.float32)


def unload_model(model: str):
    """Unload a model from Ollama to free VRAM/RAM."""
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": model, "input": "", "keep_alive": 0},
            )
        print(f"    ♻️ Unloaded model: {model}")
    except Exception as e:
        print(f"    ⚠️ Could not unload model {model}: {e}")


def get_embeddings_batch(texts: list[str], model: str, unload_after: bool = False) -> list[np.ndarray]:
    """Get embedding vectors for a batch of texts.

    Args:
        texts: List of texts to embed.
        model: Ollama model name.
        unload_after: If True, unload the model from memory after the last chunk.
    """
    results = []
    total = len(texts)
    for i, text in enumerate(texts):
        if i % 10 == 0 and i > 0:
            print(f"    📊 Embedded {i}/{total} chunks...")
        # On the last chunk with unload_after=True, set keep_alive=0
        ka = 0 if (unload_after and i == total - 1) else -1
        emb = get_embedding(text, model, keep_alive=ka)
        results.append(emb)
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def retrieve_top_k(
    query_embedding: np.ndarray,
    stored_embeddings: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve top-k most similar chunks based on cosine similarity.
    stored_embeddings: list of {"chunk_text": str, "embedding": np.ndarray, ...}
    Duplicates (same chunk_text) are removed — highest similarity is kept.
    Returns sorted list of top-k unique items with similarity scores.
    """
    scored = []
    for item in stored_embeddings:
        emb = item["embedding"]
        sim = cosine_similarity(query_embedding, emb)
        scored.append({**item, "similarity": sim})

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    # Deduplicate by chunk_text — keep first occurrence (highest similarity)
    seen: set[str] = set()
    unique = []
    for item in scored:
        text = item["chunk_text"]
        if text not in seen:
            seen.add(text)
            unique.append(item)

    return unique[:top_k]

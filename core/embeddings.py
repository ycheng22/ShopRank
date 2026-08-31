import hashlib
import os

import numpy as np

from providers.bge import BGEProvider


def _truncate_and_normalize(vector: list[float], dim: int) -> list[float]:
    vec = np.array(vector, dtype=np.float32)
    if len(vec) > dim:
        vec = vec[:dim]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

# Lazy loaded provider to avoid loading model if not needed
_provider = None

def get_provider():
    global _provider
    if _provider is None:
        _provider = BGEProvider()
    return _provider

def get_cache_key(text: str, dim: int) -> str:
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return f"bge-m3_{dim}_{h}"

async def embed_documents(texts: list[str], dim: int = 768, cache_dir: str = ".cache/embeddings") -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    provider = get_provider()
    
    results = []
    texts_to_embed = []
    indices_to_embed = []
    
    for i, text in enumerate(texts):
        cache_path = os.path.join(cache_dir, f"{get_cache_key(text, dim)}.npy")
        if os.path.exists(cache_path):
            results.append(np.load(cache_path))
        else:
            results.append(None)
            texts_to_embed.append(text)
            indices_to_embed.append(i)
            
    if texts_to_embed:
        raw_embeddings = await provider.embed(texts_to_embed, dim=dim)
        for i, text, raw_emb in zip(indices_to_embed, texts_to_embed, raw_embeddings):
            emb = _truncate_and_normalize(raw_emb, dim)
            emb_np = np.array(emb, dtype=np.float32)
            results[i] = emb_np
            
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cache_path = os.path.join(cache_dir, f"bge-m3_{dim}_{text_hash}.npy")
            np.save(cache_path, emb_np)
            
    return np.stack(results)

async def embed_query(text: str, dim: int = 768, cache_dir: str = ".cache/embeddings") -> np.ndarray:
    docs = await embed_documents([text], dim=dim, cache_dir=cache_dir)
    return docs[0]

from typing import List

class BGEProvider:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("BAAI/bge-m3", device=device, cache_folder="D:/huggingface_cache/hub")

    async def embed(self, texts: list[str], dim: int) -> list[list[float]]:
        # SentenceTransformer encode
        embeddings = self.model.encode(texts, batch_size=len(texts), normalize_embeddings=False)
        return embeddings.tolist()


class BGEProvider:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("BAAI/bge-m3", device=device, cache_folder="D:/huggingface_cache/hub")

    async def embed(self, texts: list[str], dim: int) -> list[list[float]]:
        # SentenceTransformer encode
        embeddings = self.model.encode(texts, batch_size=len(texts), normalize_embeddings=False)
        return embeddings.tolist()

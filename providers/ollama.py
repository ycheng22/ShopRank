import httpx

class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def embed(self, texts: list[str], dim: int) -> list[list[float]]:
        # For batch embedding, we use the /api/embed endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": "bge-m3",
                    "input": texts
                },
                timeout=300.0
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

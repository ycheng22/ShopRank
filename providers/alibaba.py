"""Alibaba Model Studio (Qwen) LLM provider for translation.

Uses the DashScope OpenAI-compatible chat API.
All configuration is injected — no environment reads.
"""

import httpx


class AlibabaProvider:
    """Chat-based translation provider using Alibaba Model Studio (Qwen)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        if not api_key:
            raise ValueError("Qwen/DashScope API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = "qwen-plus"

    async def translate(self, text: str, target_lang: str) -> str:
        """Translate text to target language via chat completion.

        Args:
            text: Source text to translate.
            target_lang: Target language code (e.g. 'zh', 'fr').

        Returns:
            Translated text string.

        Raises:
            httpx.HTTPStatusError: On API errors.
            ValueError: On empty/invalid responses.
        """
        lang_names = {"zh": "Chinese", "fr": "French", "en": "English"}
        target_name = lang_names.get(target_lang, target_lang)

        system_prompt = (
            f"You are a professional translator. Translate the following text to {target_name}. "
            "Output ONLY the translation, nothing else. No explanations, no notes, no quotes."
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 512,
                },
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Alibaba/Qwen returned empty choices")

        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("Alibaba/Qwen returned empty content")

        return content

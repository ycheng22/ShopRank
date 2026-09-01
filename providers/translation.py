"""Translation router — selects the low-cost provider for offline translation.

All LLM calls for translation go through this module. Scripts never call
a provider directly. Configuration is injected; no environment reads.
"""

from __future__ import annotations

import logging
from typing import Literal

from providers.alibaba import AlibabaProvider
from providers.deepseek import DeepSeekProvider

logger = logging.getLogger(__name__)

ProviderName = Literal["deepseek", "alibaba"]


def _build_provider(
    name: ProviderName,
    *,
    deepseek_api_key: str = "",
    deepseek_base_url: str = "https://api.deepseek.com/v1",
    qwen_api_key: str = "",
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> DeepSeekProvider | AlibabaProvider:
    if name == "deepseek":
        return DeepSeekProvider(api_key=deepseek_api_key, base_url=deepseek_base_url)
    elif name == "alibaba":
        return AlibabaProvider(api_key=qwen_api_key, base_url=qwen_base_url)
    else:
        raise ValueError(f"Unknown translation provider: {name}")


async def translate_text(
    text: str,
    target_lang: str,
    provider_name: ProviderName,
    *,
    deepseek_api_key: str = "",
    deepseek_base_url: str = "https://api.deepseek.com/v1",
    qwen_api_key: str = "",
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    fallback_provider: ProviderName | None = None,
    fallback_deepseek_api_key: str = "",
    fallback_deepseek_base_url: str = "https://api.deepseek.com/v1",
    fallback_qwen_api_key: str = "",
    fallback_qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> str:
    """Translate a single text using the specified provider with optional fallback.

    Args:
        text: Source text to translate.
        target_lang: Target language code (e.g. 'zh', 'fr').
        provider_name: Primary provider to use.
        fallback_provider: Optional fallback provider name.

    Returns:
        Translated text.

    Raises:
        Exception: If both primary and fallback fail.
    """
    provider = _build_provider(
        provider_name,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        qwen_api_key=qwen_api_key,
        qwen_base_url=qwen_base_url,
    )

    try:
        return await provider.translate(text, target_lang)
    except Exception as e:
        logger.warning(
            "Primary provider %s failed: %s",
            provider_name,
            str(e),
        )
        if fallback_provider is None:
            raise

        logger.info("Falling back to %s", fallback_provider)
        fb_provider = _build_provider(
            fallback_provider,
            deepseek_api_key=fallback_deepseek_api_key or deepseek_api_key,
            deepseek_base_url=fallback_deepseek_base_url,
            qwen_api_key=fallback_qwen_api_key or qwen_api_key,
            qwen_base_url=fallback_qwen_base_url,
        )
        return await fb_provider.translate(text, target_lang)

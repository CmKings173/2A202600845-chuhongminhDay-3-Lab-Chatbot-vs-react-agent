"""
Provider Factory — creates the right LLMProvider from environment variables.

Usage:
    from src.core.provider_factory import create_provider
    llm = create_provider()
"""

import os
from src.core.llm_provider import LLMProvider


def create_provider() -> LLMProvider:
    """
    Read DEFAULT_PROVIDER from .env and return the matching LLMProvider.

    Supported values: openai | google | local
    """
    provider = os.getenv("DEFAULT_PROVIDER", "openai").lower()

    if provider == "openai":
        from src.core.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model_name=os.getenv("DEFAULT_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "google":
        from src.core.gemini_provider import GeminiProvider
        return GeminiProvider(
            model_name=os.getenv("DEFAULT_MODEL", "gemini-1.5-flash"),
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    elif provider == "local":
        from src.core.local_provider import LocalProvider
        model_path = os.getenv(
            "LOCAL_MODEL_PATH",
            "./models/Phi-3-mini-4k-instruct-q4.gguf",
        )
        return LocalProvider(model_path=model_path)

    else:
        raise ValueError(
            f"Unknown DEFAULT_PROVIDER='{provider}'. "
            "Valid options: openai | google | local"
        )

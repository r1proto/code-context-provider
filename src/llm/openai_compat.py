"""
Minimal OpenAI-compatible helper utilities.

- configure_openai_compat(api_key, base_url)
    Sets environment variables commonly used by OpenAI-compatible SDKs and
    returns a dict suitable for passing to OpenAIProvider(**provider_kwargs).

- OpenAICompatClient (optional)
    A small aiohttp-based client for direct chat/embeddings calls to OpenAI-compatible
    endpoints. aiohttp is optional; use this if you need direct HTTP calls outside of
    pydantic_ai providers.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

# Optional async client depends on aiohttp. Import lazily to avoid hard dependency.
try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover - aiohttp optional
    aiohttp = None  # type: ignore


def configure_openai_compat(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Ensure environment variables and a provider_kwargs dict for OpenAI-compatible endpoints.

    - api_key: bearer API key to use (sets OPENAI_API_KEY)
    - base_url: base URL for the provider (sets OPENAI_BASE_URL, OPENAI_API_BASE, OPENAI_API_BASE_URL)

    Returns a provider_kwargs dict you can pass to OpenAIProvider(**provider_kwargs).
    """
    provider_kwargs: Dict[str, Any] = {}

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        # Some codepaths / libraries may look for alternate env names
        os.environ["OPENAI_API_KEY_OVERRIDE"] = api_key
        provider_kwargs["api_key"] = api_key

    if base_url:
        # normalize
        base_url = base_url.rstrip("/")
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_API_BASE_URL"] = base_url
        provider_kwargs["base_url"] = base_url

    return provider_kwargs


# Optional lightweight async client for direct calls if needed.
class OpenAICompatClient:
    """
    Minimal async client to call OpenAI-compatible endpoints directly.

    Requires aiohttp to be installed. Methods return parsed JSON responses.

    Example:
        client = OpenAICompatClient(api_key="sk-...", base_url="https://api.openai.com")
        await client.chat_completion({"model": "gpt-4o-mini", "messages": [...]})
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com", timeout: int = 60) -> None:
        if aiohttp is None:  # pragma: no cover - runtime optional
            raise RuntimeError("aiohttp is required for OpenAICompatClient. Install aiohttp.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session: Optional["aiohttp.ClientSession"] = None

    async def _ensure_session(self) -> "aiohttp.ClientSession":
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    async def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"OpenAI-compatible API error {resp.status}: {text}")
            return json.loads(text)

    async def embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with session.post(url, json=payload, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"OpenAI-compatible API error {resp.status}: {text}")
            return json.loads(text)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

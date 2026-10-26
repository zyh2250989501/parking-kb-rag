import hashlib
import json
from pathlib import Path
from typing import Protocol


class LLMProvider(Protocol):
    def complete(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str: ...


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=60.0, max_retries=2
        )

    def complete(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class CachedProvider:
    def __init__(self, inner: LLMProvider, cache_dir: Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self, messages: list[dict[str, str]], temperature: float
    ) -> str:
        model = getattr(self.inner, "model", type(self.inner).__name__)
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str:
        path = self.cache_dir / self._cache_key(messages, temperature)
        if path.exists():
            return path.read_text(encoding="utf-8")
        text = self.inner.complete(messages, temperature)
        path.write_text(text, encoding="utf-8")
        return text

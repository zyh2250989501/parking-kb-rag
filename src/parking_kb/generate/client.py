import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Protocol


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


def _is_rate_limited(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "rate limit" in str(exc).lower()


class RetryingProvider:
    """限流退避：TPM 的窗口是一分钟，SDK 自带的两次重试熬不过去。

    长跑评估里一次 429 会让整轮结果作废，所以在这里等够时间再试。
    """

    def __init__(
        self,
        inner: LLMProvider,
        attempts: int = 6,
        base_delay: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay
        self._sleep = sleep

    def complete(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str:
        for attempt in range(self.attempts):
            try:
                return self.inner.complete(messages, temperature)
            except Exception as exc:
                if not _is_rate_limited(exc) or attempt == self.attempts - 1:
                    raise
                self._sleep(self.base_delay * (attempt + 1))
        raise AssertionError("unreachable")


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

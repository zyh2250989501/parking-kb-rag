import pytest

from parking_kb.generate.client import RetryingProvider


class _RateLimited(Exception):
    status_code = 429


class _FlakyProvider:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def complete(self, messages, temperature=0.0):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _RateLimited("429 rate limit")
        return "ok"


def test_retrying_provider_waits_out_rate_limit():
    """长跑评估里一次 TPM 限流不该让几十分钟的结果作废。"""
    slept: list[float] = []
    provider = RetryingProvider(
        _FlakyProvider(fail_times=2), sleep=slept.append, base_delay=10.0
    )

    assert provider.complete([{"role": "user", "content": "x"}]) == "ok"
    assert slept == [10.0, 20.0]


def test_retrying_provider_gives_up_after_attempts():
    slept: list[float] = []
    provider = RetryingProvider(
        _FlakyProvider(fail_times=99), attempts=3, sleep=slept.append, base_delay=1.0
    )

    with pytest.raises(_RateLimited):
        provider.complete([{"role": "user", "content": "x"}])
    assert len(slept) == 2


def test_retrying_provider_does_not_swallow_other_errors():
    class _Boom:
        def complete(self, messages, temperature=0.0):
            raise ValueError("bad json")

    with pytest.raises(ValueError):
        RetryingProvider(_Boom(), sleep=lambda seconds: None).complete([])

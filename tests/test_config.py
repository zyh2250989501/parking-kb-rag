import pytest
from pydantic import ValidationError

from parking_kb.config import load_settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    s = load_settings(env_file=None)
    assert s.deepseek_model == "deepseek-chat"
    assert s.data_dir.name == "data"


def test_settings_requires_deepseek_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        load_settings(env_file=None)

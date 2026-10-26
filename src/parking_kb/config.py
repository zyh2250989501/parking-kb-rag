from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    judge_model: str = "Qwen/Qwen2.5-72B-Instruct"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    data_dir: Path = Path("data")
    corpus_dir: Path = Path("corpus")
    device: str = "cuda"


def load_settings(env_file: Path | None = None) -> Settings:
    return Settings(_env_file=env_file)

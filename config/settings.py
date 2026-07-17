from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and `.env`."""

    # Telegram MTProto
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_session_string: str = ""
    telegram_phone: str = ""

    # OpenAI (LLM + embeddings)
    openai_api_key: str = ""

    # Alert delivery
    pantheon_bot_token: str = ""
    pantheon_chat_id: int = 0
    eidolon_bot_token: str = ""  # Dedicated Eidolon bot (falls back to Pantheon)

    # Paths
    db_path: Path = Path("data/eidolon.db")
    watchers_path: Path = Path("config/watchers.yml")

    # Embedding filter (Level 2)
    embedding_model: str = "text-embedding-3-small"
    chroma_path: Path = Path("data/chroma")

    # LLM filtering (Level 3)
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_seconds: int = 15

    # Daily summary
    summary_enabled: bool = True
    summary_hour_utc: int = 13  # 20:00 ICT (UTC+7)
    summary_model: str = "gpt-4.1-mini"

    # Debug
    debug_echo: bool = False  # Forward ALL messages from monitored chats

    # Processing
    batch_size: int = 50
    batch_interval_seconds: int = 300  # 5 min
    embedding_similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    embedding_negative_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    processing_queue_size: int = Field(default=500, ge=1, le=100_000)
    processing_workers: int = Field(default=4, ge=1, le=64)
    shutdown_timeout_seconds: int = Field(default=30, ge=1, le=300)

    # Data minimization
    store_raw_telegram_json: bool = False
    retention_days: int = Field(default=30, ge=1, le=3650)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

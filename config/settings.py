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
    # Reconnaissance keeps its own file: bulk crawl writes must never queue
    # behind live alert delivery on the monitoring connection.
    scout_db_path: Path = Path("data/eidolon_scout.db")
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
    embedding_similarity_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    # A small negative value makes Level 2 recall-oriented: the positive
    # reference may trail a hard negative slightly before the LLM resolves it.
    embedding_negative_margin: float = Field(default=-0.05, ge=-1.0, le=1.0)
    processing_queue_size: int = Field(default=500, ge=1, le=100_000)
    processing_workers: int = Field(default=4, ge=1, le=64)
    ingress_max_attempts: int = Field(default=3, ge=1, le=10)
    ingress_retry_base_seconds: float = Field(default=0.25, ge=0.05, le=10.0)
    shutdown_timeout_seconds: int = Field(default=30, ge=1, le=300)
    outbox_poll_seconds: float = Field(default=0.5, ge=0.1, le=60.0)
    outbox_batch_size: int = Field(default=50, ge=1, le=1000)
    # One Telegram delivery cycle is bounded by three 10-second requests and
    # two server-directed waits of up to 60 seconds. Keep the lease above that
    # ceiling so another worker cannot reclaim an in-flight alert.
    outbox_lease_seconds: int = Field(default=180, ge=180, le=3600)
    outbox_max_attempts: int = Field(default=5, ge=1, le=20)
    recovery_batch_size: int = Field(default=500, ge=1, le=10_000)

    # Data minimization
    store_raw_telegram_json: bool = False
    retention_days: int = Field(default=30, ge=1, le=3650)
    retention_sweep_hours: int = Field(default=24, ge=1, le=168)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Telegram MTProto
    telegram_api_id: int
    telegram_api_hash: str
    telegram_session_string: str = ""
    telegram_phone: str = ""

    # OpenAI (LLM + embeddings)
    openai_api_key: str = ""

    # Alert delivery
    pantheon_bot_token: str = ""
    pantheon_chat_id: int = 60972166
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
    embedding_similarity_threshold: float = 0.70

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

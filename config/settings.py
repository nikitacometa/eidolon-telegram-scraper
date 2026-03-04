from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Telegram MTProto
    telegram_api_id: int
    telegram_api_hash: str
    telegram_session_string: str = ""
    telegram_phone: str = ""

    # LLM APIs
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Alert delivery (Pantheon bot)
    pantheon_bot_token: str = ""
    pantheon_chat_id: int = 60972166

    # Paths
    db_path: Path = Path("data/eidolon.db")
    watchers_path: Path = Path("config/watchers.yml")

    # Processing
    batch_size: int = 50
    batch_interval_seconds: int = 300  # 5 min
    embedding_similarity_threshold: float = 0.70

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

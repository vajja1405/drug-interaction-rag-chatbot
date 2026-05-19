"""
config.py — Centralised application settings.

Reads configuration from environment variables (or a .env file via python-dotenv).
All modules import from here so that changing a setting only requires editing one place.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="sk-placeholder", env="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", env="OPENAI_BASE_URL"
    )
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(
        default=0.1,
        description="Low temperature for clinical accuracy",
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens to generate (critical for local models to prevent truncation)",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
    )

    # ── Vector store ──────────────────────────────────────────────────────────
    vector_store_path: str = Field(
        default="data/vectorstore", env="VECTOR_STORE_PATH"
    )
    rag_top_k: int = Field(default=5, env="RAG_TOP_K")
    similarity_threshold: float = Field(default=0.55, env="SIMILARITY_THRESHOLD")

    # ── External APIs ─────────────────────────────────────────────────────────
    rxnorm_base_url: str = Field(
        default="https://rxnav.nlm.nih.gov/REST", env="RXNORM_BASE_URL"
    )
    openfda_base_url: str = Field(
        default="https://api.fda.gov/drug/label.json", env="OPENFDA_BASE_URL"
    )

    # ── Caching ───────────────────────────────────────────────────────────────
    redis_url: str | None = Field(default=None, env="REDIS_URL")

    # ── Data paths ────────────────────────────────────────────────────────────
    raw_data_path: str = Field(default="data/raw", env="RAW_DATA_PATH")
    processed_data_path: str = Field(
        default="data/processed", env="PROCESSED_DATA_PATH"
    )

    # ── API ───────────────────────────────────────────────────────────────────
    max_drugs_per_request: int = Field(default=20, env="MAX_DRUGS_PER_REQUEST")
    rate_limit_per_minute: int = Field(default=30, env="RATE_LIMIT_PER_MINUTE")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton – import this everywhere
settings = Settings()

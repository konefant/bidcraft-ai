"""
Configuration centrale de l'application.

Rôle : charger et valider les variables d'environnement (fichier .env)
au démarrage. Si une variable obligatoire manque (ex. OPENAI_API_KEY),
l'application refuse de démarrer avec une erreur claire, plutôt que de
planter plus tard en production.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres de l'application, lus depuis l'environnement / le fichier .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore les variables .env inconnues
    )

    # --- Application -------------------------------------------------------
    APP_NAME: str = "BidCraft AI API"
    DEBUG: bool = False

    # --- CORS : origines autorisées (frontend Next.js) -----------------------
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- OpenAI --------------------------------------------------------------
    OPENAI_API_KEY: str
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_TEMPERATURE: float = 0.3  # faible = rédaction plus factuelle et stable

    # --- Base de données vectorielle (pgvector / Supabase) -------------------
    # Optionnelle : si absente, le module RAG fonctionne en mode "simulation".
    DATABASE_URL: str | None = None
    VECTOR_COLLECTION_NAME: str = "bidcraft_knowledge"

    # --- Découpage du texte (chunking) ---------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # --- Upload ---------------------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 20


@lru_cache
def get_settings() -> Settings:
    """Retourne une instance unique (mise en cache) des paramètres."""
    return Settings()  # type: ignore[call-arg]


settings: Settings = get_settings()
import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    CHROMA_HOST: str
    CHROMA_PORT: str
    COLLECTION: str
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    DB_URL: str
    API_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix="",
    )


CORE_FOLDER = pathlib.Path(__file__).parent
APP_FOLDER = CORE_FOLDER.parent
PROJECT_FOLDER = APP_FOLDER.parent

settings = Settings(_env_file=PROJECT_FOLDER / ".env")

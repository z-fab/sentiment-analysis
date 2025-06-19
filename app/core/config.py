import pathlib

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CHROMA_HOST: str
    CHROMA_PORT: str
    COLLECTION: str
    OPENAI_API_KEY: str
    DB_URL: str


CORE_FOLDER = pathlib.Path(__file__).parent
APP_FOLDER = CORE_FOLDER.parent
PROJECT_FOLDER = APP_FOLDER.parent

settings = Settings(_env_file=PROJECT_FOLDER / ".env")

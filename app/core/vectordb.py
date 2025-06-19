import chromadb
from chromadb.config import Settings
from core.config import settings

CHROMA_CLIENT = client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True),
)

COLLECTION = CHROMA_CLIENT.get_or_create_collection(settings.COLLECTION)

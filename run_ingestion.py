from src.components.loaders import LocalDirLoader
from src.components.splitters import TextSplitter
from src.components.vector_db import RagDBWrapper
from src.pipelines.ingestion import RAGingestion

from config.settings import settings

def create_db():
    loader_docs = LocalDirLoader(settings.SOURCE_DIR)
    split_docs = TextSplitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    vector_db = RagDBWrapper(settings.EMBEDDING_MODEL, settings.COLLECTION_NAME)
    
    create_data = RAGingestion(loader_docs, split_docs, vector_db)
    create_data.run()

if __name__ == "__main__":
    create_db()
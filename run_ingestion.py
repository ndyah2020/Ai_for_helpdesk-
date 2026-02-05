from src.components.loaders import LocalDirLoader
from src.components.splitters import TextSplitter
from src.database.qdrant_db import QdrantDBWrapper
from src.pipelines.ingestion import RAGingestion
from config.settings import settings

class RunIngestion: 
    def __init__(self):
        loader = LocalDirLoader(settings.SOURCE_DIR)
        splitter = TextSplitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        vector_db = QdrantDBWrapper(settings.EMBEDDING_MODEL, settings.COLLECTION_NAME)
        
        self.pipeline = RAGingestion(loader, splitter, vector_db)
    
    def create_db(self):
        self.pipeline.run()

    def reset_db(self):
        self.pipeline.reset_db()

if __name__ == "__main__":
    run_ingestion = RunIngestion()
    run_ingestion.create_db()
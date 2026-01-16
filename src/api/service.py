import logging
from src.components.vector_db import RagDBWrapper
from src.components.llm import OllamaRAGLLM
from src.components.reranker import HuggingFaceReranker
from src.pipelines.rag import RAGpipeline
from config.settings import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.pipeline = None

    def initialize(self):
        if(self.pipeline):
            return
    
        logger.info("Đang khởi tạo RAG Service...")

        try:
            vector_db = RagDBWrapper(settings.EMBEDDING_MODEL, settings.COLLECTION_NAME)
            # reranker = HuggingFaceReranker(settings.CROSS_ENCODER, settings.TOP_CHUNK)
            llm = OllamaRAGLLM()
            
            self.pipeline = RAGpipeline(vector_db, llm)
            logger.info("RAG Service sẵn sàng!")
        except Exception as e:
            logger.error("Lỗi Khởi tạo RAG")
            raise e

    def get_answer(self, input: str):
            if not self.pipeline:
                self.initialize()
            return self.pipeline.run(input)


rag_service = RAGService()
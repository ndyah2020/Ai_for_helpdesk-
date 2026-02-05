import logging
from src.database.qdrant_db import QdrantDBWrapper
from src.components.llm import OllamaRAGLLM
from src.components.reranker import HuggingFaceReranker
from src.components.loaders import LocalDirLoader
from src.pipelines.rag import RAGpipeline
from config.settings import settings

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.pipeline = None
        self.vector_db = None

    def initialize(self):
        if self.pipeline:
            return
    
        logger.info("Đang khởi tạo RAG Service...")

        try:
            self.vector_db = QdrantDBWrapper(settings.EMBEDDING_MODEL, settings.COLLECTION_NAME)
            
            # Khởi tạo BM25 cho Hybrid Search
            # Load file vào RAM để đánh index từ khóa (Chỉ làm 1 lần lúc start server)
            logger.info("Đang tải dữ liệu để xây dựng chỉ mục từ khóa (BM25)...")
            try:
                loader = LocalDirLoader(settings.SOURCE_DIR)
                all_docs = loader.load_documents() 
                
                if all_docs:
                    self.vector_db.init_bm25(all_docs)
                    logger.info(f"Đã index BM25 thành công cho {len(all_docs)} chunks.")
                else:
                    logger.warning("Không tìm thấy tài liệu nào. Hệ thống sẽ bỏ qua BM25.")
            except Exception as e:
                # Nếu lỗi BM25 thì vẫn chạy tiếp, chỉ dùng Vector Search thôi
                logger.warning(f"Lỗi khởi tạo BM25 (Không nghiêm trọng): {e}")

            # 3. Khởi tạo Reranker & LLM
            reranker = HuggingFaceReranker(settings.CROSS_ENCODER, settings.TOP_CHUNK)
            llm = OllamaRAGLLM()
            
            # 4. Ráp vào Pipeline
            self.pipeline = RAGpipeline(self.vector_db, reranker, llm)

            logger.info("RAG Service đã sẵn sàng phục vụ!")
            
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi khởi tạo RAG: {e}")
            raise e

    def get_answer(self, input: str, k: int = None, mmr_diversity: float = 0.5) -> str:
        if not self.pipeline:
            self.initialize()
        return self.pipeline.run(input, k=k, mmr_diversity=mmr_diversity)
    
rag_service = RAGService()
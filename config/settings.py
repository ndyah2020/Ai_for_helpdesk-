from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # nơi lấy file docs
    SOURCE_DIR: str = "data/source_documents"
    # nơi lưu trữ vector data
    CHROMA_DB_DIR: str = "data/persistent_chroma_db"
    COLLECTION_NAME: str = "my_rag_collection"
    # độ sáng tạo llm
    TEMPERATURE: float = 0.0
    # đoạn văn bảng được cắt
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    # số chunk được lấy ra từ chroma data
    CHUNK_NUMBER: int = 5
    # model llm để trả lời
    MODEL_NAME: str = "gpt-oss:120b"
    # model embedding
    EMBEDDING_MODEL: str = "bge-m3"
    # Top chunk sau khi qua đánh giá
    TOP_CHUNK: int = 5
    # đăng ký blueprint
    URL_PREFIX: str = "/api/v1"
settings = Settings()
        
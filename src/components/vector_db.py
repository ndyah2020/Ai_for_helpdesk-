from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.interfaces import BaseVectorDB
from config.settings import settings
from langchain_postgres import PGVector
from dotenv import load_dotenv
import os

class RagDBWrapper(BaseVectorDB):
    def __init__(self, embeddings_model: str, collection_name: str):
        load_dotenv()
        self.embedding_function = OllamaEmbeddings(
            model=embeddings_model,
            base_url=os.getenv('EMBEDDING_BASE_URL')
        )
        
        self.vector_store = PGVector(
            embeddings=self.embedding_function,
            collection_name=collection_name,
            connection=os.getenv('POSTGRE_CONNECTION'),         
            use_jsonb=True,
        )
    
    def add_documents(self, documents : List[Document]):
        print("Đang thêm tài liệu vào PostGreSQL...")
        try:
            self.vector_store.add_documents(documents)
        except Exception as e:
            print("Dữ liệu này đã tồn tại, bỏ qua hoặc cần xử lý update thủ công.")

        print(f"Đã lưu dữ liệu vào thư mục: {settings.CHROMA_DB_DIR}")

    def reset_db(self):
        self.vector_store.drop_tables() 
        self.vector_store.delete_collection()
        print("Đã xóa sạch database!")

    def get_retriever(self, k: int):
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
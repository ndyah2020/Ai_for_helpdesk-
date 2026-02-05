from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.interfaces import BaseVectorDB
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

class QdrantDBWrapper(BaseVectorDB):
    def __init__(self, embeddings_model: str, collection_name: str):
        load_dotenv()
        self.embedding_function = OllamaEmbeddings(model=embeddings_model, base_url=os.getenv('EMBEDDING_BASE_URL'))
        self.collection_name = collection_name
        self.url=os.getenv("QDRANT_URL")
        self.api_key_cloud=os.getenv("QDRANT_API")
        
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key_cloud,
            timeout=60
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedding_function,
        )
    
    def add_documents(self, documents : List[Document]):
        print(f"Đang upload {len(documents)} tài liệu lên Cloud...")
        try:
            self.vector_store.add_documents(documents)
            print("Upload thành công!")
        except Exception as e:
            print(f"Lỗi khi upload: {e}")

    def _ensure_collection_exists(self):
        """Tạo collection và đánh Index cho trường 'source'"""
        if not self.client.collection_exists(collection_name=self.collection_name):
            print(f"Collection '{self.collection_name}' chưa có. Đang tạo mới...")
            
            # 1. Lấy kích thước vector (1024 cho bge-m3)
            dummy_vec = self.embedding_function.embed_query("test")
            vec_size = len(dummy_vec)
            
            # 2. Tạo Collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vec_size, 
                    distance=models.Distance.COSINE
                )
            )
            print(f"Đã tạo Collection (Size: {vec_size})")

            # 3. TẠO INDEX CHO TRƯỜNG 'source
            print("Đang tạo Payload Index cho field 'source'...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print("Đã tạo Index thành công!")
        
        # Trường hợp Collection đã có nhưng chưa có Index
        # Đoạn này chạy thừa cũng không sao, Qdrant sẽ tự bỏ qua nếu đã có index
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    def reset_db(self):
        # Reset: Xóa sạch và tạo lại cáim mới
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Đã xóa collection cũ: {self.collection_name}")
            self._ensure_collection_exists()
            
            print("Database đã được reset và sẵn sàng nạp mới!")
        except Exception as e:
            # Nếu collection chưa tồn tại mà lỡ gọi xóa thì bỏ qua lỗi
            print(f"Lỗi reset (có thể collection chưa từng tồn tại): {e}")
            self._ensure_collection_exists()

    def delete_file_data(self, file_path: str):
        print(f"Đang xóa dữ liệu cũ của file: {file_path}...")
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=file_path),
                            ),
                        ],
                    )
                ),
            )
            print("Đã xóa xong dữ liệu cũ.")
        except Exception as e:
            print(f"Lỗi khi xóa file cũ (có thể file chưa từng tồn tại): {e}")
    
    def get_retriever(self, k: int, search_type: str = "similarity"):
        kwargs = {"k": k}
        if search_type == "mmr":
            kwargs["fetch_k"] = k * 4
            kwargs["lambda_mult"] = 0.5

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=kwargs
        )
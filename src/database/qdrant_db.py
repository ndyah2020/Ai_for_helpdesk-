from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.interfaces import BaseVectorDB
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import os

class QdrantDBWrapper(BaseVectorDB):
    def __init__(self, embeddings_model: str, collection_name: str):
        load_dotenv()
        self.embedding_function = OllamaEmbeddings(
            model=embeddings_model, 
            base_url=os.getenv('EMBEDDING_BASE_URL')
        )

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API"),
            timeout=100
        )

        self.collection_name = collection_name
        self._ensure_collection_exists()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedding_function,
        )

        self.bm25_retriever = None

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
            print("Đang tạo Payload Index cho field 'metadata.source'...")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print("Đã tạo Index thành công!")

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    def reset_db(self):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Đã xóa collection cũ: {self.collection_name}")
            self._ensure_collection_exists()
            
            print("Database đã được reset và sẵn sàng nạp mới!")
        except Exception as e:
            print(f"Lỗi reset (có thể collection chưa từng tồn tại): {e}")
            self._ensure_collection_exists()

    def delete_file_data(self, file_path: str):
        print(f"Đang yêu cầu xóa file với chuỗi: '{file_path}'")
        
        try:
            sample, _ = self.client.scroll(collection_name=self.collection_name, limit=1)
            if sample:
                print("\n[DEBUG] MỘT CHUNK TRONG DB ĐANG CÓ PAYLOAD NHƯ SAU:")
                print(sample[0].payload)
                print("-" * 40)
        except Exception as e:
            pass
        # ---------------------------------------------------

        # Cấu hình filter (langchain_qdrant lưu metadata trong key 'metadata')
        file_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source", 
                    match=models.MatchValue(value=file_path),
                ),
            ]
        )

        try:
            # Bắt buộc đếm xem có trúng chunk nào không
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=file_filter,
                exact=True
            )
            
            if count_result.count == 0:
                print(f"THẤT BẠI: Qdrant không tìm thấy chunk nào có metadata.source chính xác là '{file_path}'")
                print("Vui lòng so sánh chuỗi bạn truyền vào với kết quả [DEBUG] ở trên xem có khác nhau dấu gạch chéo (/, \\) hay đường dẫn tuyệt đối/tương đối không.")
                return

            # Gọi lệnh xóa VÀ BẮT BUỘC CHỜ XÓA XONG (wait=True)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=file_filter),
                wait=True  # Bắt buộc phải có để tránh đụng độ với add_documents
            )
            print(f"✅ Đã xóa THÀNH CÔNG {count_result.count} chunks của file này.")
            
        except Exception as e:
            print(f"Lỗi hệ thống khi xóa file: {e}")
            
    def init_bm25(self, all_documents: List[Document]):
        print("Đang đánh index BM25 (Keyword Search)...")
        self.bm25_retriever = BM25Retriever.from_documents(all_documents)
        print("BM25 đã sẵn sàng!")

    def get_retriever(self, k: int = 4, mmr_diversity: float = 0.5):
        # 1. Cấu hình Qdrant với MMR (Dense + Diversity)
        # MMR giúp loại bỏ các vector quá giống nhau, tăng tính đa dạng
        qdrant_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k, 
                "fetch_k": k * 4,     
                "lambda_mult": 1 - mmr_diversity 
            }
        )
        # 2. Nếu chưa có BM25 , chỉ trả về Qdrant MMR
        if not self.bm25_retriever:
            print("Chưa khởi tạo BM25, chỉ sử dụng Qdrant Vector Search.")
            return qdrant_retriever

        # BM25Retriever cho phép chỉnh k trực tiếp
        self.bm25_retriever.k = k

        # Tỷ lệ 50-50 cho Semantic và Keyword
        ensemble_retriever = EnsembleRetriever(
            retrievers=[qdrant_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        return ensemble_retriever
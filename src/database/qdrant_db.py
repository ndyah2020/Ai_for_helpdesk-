from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.interfaces import BaseVectorDB
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

class QdrantDBWrapper(BaseVectorDB):
    def __init__(self, embeddings_model: str, collection_name: str):
        load_dotenv()
        self.embedding_function = OllamaEmbeddings(model=embeddings_model, base_url=os.getenv('EMBEDDING_BASE_URL'))

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

    def reset_db(self):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Đã xóa collection trên Cloud: {self.collection_name}")
        except Exception as e:
            print(f"Lỗi khi reset DB: {e}")

    def get_retriever(self, k: int, search_type: str = "similarity"):
        kwargs = {"k": k}
        if search_type == "mmr":
            kwargs["fetch_k"] = k * 4
            kwargs["lambda_mult"] = 0.5

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=kwargs
        )
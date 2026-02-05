from src.interfaces import BaseLoader, BaseSplitter, BaseVectorDB

class RAGingestion:
    def __init__(self, loader_docs: BaseLoader, split_docs: BaseSplitter, vector_db: BaseVectorDB):
        self.loader_docs = loader_docs
        self.split_docs = split_docs
        self.vector_db = vector_db

    def reset_db(self):
        self.vector_db.reset_db(self)
         
    def run(self):
        print("Bắt đầu xử lý tài liệu PDF...")
        docs = self.loader_docs.load_documents()
        print(f"Đã tải {len(docs)} trang PDF.")
        split_docs = self.split_docs.split_documents(docs)
        print(f"Đã chia thành {len(split_docs)} đoạn văn bản (chunks).")
        self.vector_db.add_documents(split_docs)
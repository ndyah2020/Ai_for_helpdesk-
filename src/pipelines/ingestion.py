from src.interfaces import BaseLoader, BaseSplitter, BaseVectorDB
import os
from tqdm import tqdm
from src.untils.file_tracker import FileTracker

class RAGingestion:
    def __init__(self, loader_docs: BaseLoader, split_docs: BaseSplitter, vector_db: BaseVectorDB):
        self.loader_docs = loader_docs
        self.split_docs = split_docs
        self.vector_db = vector_db
        self.tracker = FileTracker()

    def reset_db(self):
        self.vector_db.reset_db()
        if os.path.exists(self.tracker.tracking_file):
            os.remove(self.tracker.tracking_file)    
            self.tracker = FileTracker()
         
    def run(self):
        print("Bắt đầu quy trình Smart Ingestion...")
        
        file_paths = self.loader_docs.get_file_paths()
        
        if not file_paths:
            print("Không tìm thấy file nào!")
            return

        print(f"Tìm thấy {len(file_paths)} file. Đang kiểm tra thay đổi...")

        # 2. Duyệt từng file
        files_processed = 0
        for file_path in file_paths:
            # Chuẩn hóa path
            file_path = os.path.normpath(file_path)
            file_name = os.path.basename(file_path)

            # A. Kiểm tra Hash (Có thay đổi không?)
            is_changed, new_hash = self.tracker.check_file_status(file_path)

            if not is_changed:
                # print(f"Skipping: {file_name}")
                continue
            
            # === NẾU CÓ THAY ĐỔI ===
            print(f"Đang cập nhật: {file_name}...")
            
            try:
                self.vector_db.delete_file_data(file_path)

                # Load nội dung file đó
                docs = self.loader_docs.load_single_file(file_path)
                if not docs:
                    print(f"File rỗng: {file_name}")
                    continue

                # Split thành chunks
                chunks = self.split_docs.split_documents(docs)

                # Gán metadata chuẩn (để sau này xóa được)
                for chunk in chunks:
                    chunk.metadata["source"] = file_path

                # Nạp vào DB (Dùng hàm add_documents của DB Wrapper)
                # DB Wrapper nên có sẵn logic batching
                self.vector_db.add_documents(chunks)

                # G. Cập nhật Tracker
                self.tracker.update_status(file_path, new_hash)
                files_processed += 1

            except Exception as e:
                print(f"Lỗi xử lý file {file_name}: {e}")

        if files_processed == 0:
            print("Dữ liệu đã đồng bộ. Không có file nào mới.")
        else:
            print(f"Đã cập nhật thành công {files_processed} file!")

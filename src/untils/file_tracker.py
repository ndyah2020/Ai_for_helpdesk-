import hashlib
import json
import os
from datetime import datetime, timezone

class FileTracker:
    def __init__(self, tracking_file="file_status.json"):
        self.tracking_file = tracking_file
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_history(self):
        with open(self.tracking_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

    def get_file_hash(self, file_path):
        """Tính MD5 hash của file"""
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except FileNotFoundError:
            return None

    def check_file_status(self, file_path):
        """
        Kiểm tra trạng thái file.
        Returns: (is_changed, new_hash)
        """
        current_hash = self.get_file_hash(file_path)
        
        # Nếu file không tồn tại (đã bị xóa khỏi ổ cứng)
        if current_hash is None:
            return False, None

        file_name = os.path.basename(file_path)

        # Lấy hash từ format mới hoặc format cũ
        if file_name not in self.history:
            if file_path in self.history:
                record = self.history[file_path]
                old_hash = record if isinstance(record, str) else record.get("hash")
                if old_hash != current_hash:
                    return True, current_hash
                return False, current_hash
            
            # File là mới hoàn toàn về tên, kiểm tra xem nội dung đã tồn tại chưa
            if self.is_hash_exists(current_hash):
                # Nội dung đã tồn tại dưới tên khác -> Không cần nạp lại để tránh trùng lặp chuộng
                return False, current_hash
                
            return True, current_hash
        
        # Lấy file đã có trong history
        record = self.history[file_name]
        old_hash = record.get("hash") if isinstance(record, dict) else record

        # Nếu hash khác hash cũ -> Đã thay đổi
        if old_hash != current_hash:
            # Đối với file ĐÃ TỒN TẠI (cùng file_name), nếu nội dung (hash) thay đổi 
            # thì ta cho phép cập nhật, KHÔNG BỊ CHẶN BỞI `is_hash_exists` như file mới.
            # Vì nếu chặn, nó tìm thấy hash cũ của chính nó hoặc một bản sao, nó sẽ bỏ qua.
            return True, current_hash
            
        return False, current_hash

    def update_status(self, file_path, new_hash, status="embedded"):
        file_name = os.path.basename(file_path)
        normalized_path = file_path.replace("\\", "/")
        
        current_time_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self.history[file_name] = {
            "file_path": normalized_path,
            "hash": new_hash,
            "last_updated": current_time_iso,
            "status": status
        }
        self.save_history()

    def is_hash_exists(self, hash_to_check: str) -> bool:
        """Kiểm tra xem một hash đã tồn tại trong file_status.json hay chưa (tránh trùng lặp nội dung)."""
        for record in self.history.values():
            old_hash = record.get("hash") if isinstance(record, dict) else record
            if old_hash == hash_to_check:
                return True
        return False
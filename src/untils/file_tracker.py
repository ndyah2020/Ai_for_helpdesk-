import hashlib
import json
import os

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

        # Nếu file chưa từng được xử lý
        if file_path not in self.history:
            return True, current_hash
        
        # Nếu hash khác hash cũ -> Đã thay đổi
        if self.history[file_path] != current_hash:
            return True, current_hash
            
        return False, current_hash

    def update_status(self, file_path, new_hash):
        self.history[file_path] = new_hash
        self.save_history()
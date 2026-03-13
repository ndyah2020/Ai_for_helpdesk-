from flask import Blueprint, request, jsonify
from pydantic import ValidationError
import os
import hashlib
from werkzeug.utils import secure_filename
from src.api.service import rag_service
from src.api.schemas import ChatRequest, ChatResponse, AnswerData
from src.untils.file_tracker import FileTracker
from config.settings import settings

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        # 1. Validate dữ liệu đầu vào bằng Pydantic
        raw_data = request.get_json()
        print(raw_data)
        payload = ChatRequest(**raw_data)

        # 2. Gọi Service với các tham số Dynamic (k, mmr_diversity)
        # Nếu client không gửi k, payload.k sẽ là None -> Service tự dùng mặc định
        
        answer_text = rag_service.get_answer(
            input=payload.input,
            k=payload.k,
            mmr_diversity=payload.mmr_diversity
        )

        # 3. Trả về kết quả
        response = ChatResponse(
            status="success",
            data=AnswerData(
                input=payload.input,
                answer=answer_text
            )
        )

        return jsonify(response.model_dump()), 200
    
    except ValidationError as e:
        return jsonify({
            "status": "error",
            "message": "Dữ liệu không hợp lệ",
            "details": e.errors()
        }), 400
   
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({
            "status": "error",
            "message": "Lỗi hệ thống nội bộ",
            "error_details": str(e) 
        }), 500
# http://localhost:5202/api/v1/
@chat_bp.route('/ingest', methods=['POST'])
def ingest_endpoint():
    try:
        rag_service.run_ingestion()
        return jsonify({
            "status": "success",
            "message": "Đã hoàn tất quá trình nạp dữ liệu (Ingestion) thành công."
        }), 200
    except Exception as e:
        print(f"Server Error during Ingestion: {e}")
        return jsonify({
            "status": "error",
            "message": "Lỗi hệ thống trong quá trình nạp dữ liệu",
            "error_details": str(e)
        }), 500

@chat_bp.route('/upload', methods=['POST'])
def upload_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "Không tìm thấy file trong yêu cầu"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Tên file rỗng"}), 400

        # Đọc nội dung file để tính hash MD5 nhằm phát hiện trùng lặp
        file_content = file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        
        tracker = FileTracker()
        if tracker.is_hash_exists(file_hash):
            return jsonify({
                "status": "error",
                "message": "File đã tồn tại trong hệ thống (trùng lặp nội dung). Đã hủy upload để tránh trùng lặp chunks."
            }), 400

        # Đảm bảo thư mục lưu trữ tồn tại
        os.makedirs(settings.SOURCE_DIR, exist_ok=True)
        
        # Lưu file
        filename = secure_filename(file.filename)
        save_path = os.path.join(settings.SOURCE_DIR, filename)
        
        with open(save_path, "wb") as f:
            f.write(file_content)

        return jsonify({
            "status": "success",
            "message": f"Upload file '{filename}' thành công. Đã lưu tại {save_path}. Vui lòng gọi API /ingest để nạp dữ liệu gốc."
        }), 200

    except Exception as e:
        print(f"Server Error during File Upload: {e}")
        return jsonify({
            "status": "error",
            "message": "Lỗi hệ thống trong quá trình upload file",
            "error_details": str(e)
        }), 500

@chat_bp.route('/reset', methods=['POST'])
def reset_endpoint():
    try:
        rag_service.reset_db()
        return jsonify({
            "status": "success",
            "message": "Đã reset toàn bộ Database vector thành công."
        }), 200
    except Exception as e:
        print(f"Server Error during Reset: {e}")
        return jsonify({
            "status": "error",
            "message": "Lỗi hệ thống trong quá trình reset Database",
            "error_details": str(e)
        }), 500

@chat_bp.route('/files', methods=['GET'])
def get_files_endpoint():
    try:
        tracker = FileTracker()
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(settings.SOURCE_DIR, exist_ok=True)
        
        files_list = []
        
        # Lấy danh sách file thực tế trong thư mục
        for filename in os.listdir(settings.SOURCE_DIR):
            file_path = os.path.join(settings.SOURCE_DIR, filename)
            
            # Chỉ lấy file, bỏ qua thư mục con nếu có
            if os.path.isfile(file_path):
                # Kiểm tra xem file này đã được nạp (có trong tracker) hay chưa
                if filename in tracker.history:
                    file_info = tracker.history[filename]
                    if isinstance(file_info, str):
                        files_list.append({
                            "file_name": filename,
                            "hash": file_info,
                            "status": "embedded"
                        })
                    else:
                        info = file_info.copy()
                        info["file_name"] = filename
                        files_list.append(info)
                else:
                    # File có trong thư mục nhưng chưa được nạp (chưa có trong JSON)
                    files_list.append({
                        "file_name": filename,
                        "file_path": file_path.replace("\\", "/"),
                        "hash": None,
                        "last_updated": None,
                        "status": "pending_ingest" # Trạng thái chờ nạp
                    })
                
        return jsonify({
            "status": "success",
            "data": files_list
        }), 200
    except Exception as e:
        print(f"Server Error during get_files: {e}")
        return jsonify({
            "status": "error",
            "message": "Lỗi hệ thống khi lấy danh sách file",
            "error_details": str(e)
        }), 500
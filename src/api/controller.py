from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from src.api.service import rag_service
from src.api.schemas import ChatRequest, ChatResponse, AnswerData

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
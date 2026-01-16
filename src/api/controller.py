from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from src.api.service import rag_service
from src.api.schemas import ChatRequest, ChatResponse, AnswerData


chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        raw_data = request.get_json()
        payload = ChatRequest(**raw_data)
        answer_text = rag_service.get_answer(payload.input)

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
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@chat_bp.route('/webhook/rocketchat', methods=['POST'])
def rocketchat_webhook():
    try:
        data = request.get_json() or {}
        if data.get('bot') or data.get('user_name') == 'ragbot': 
            return jsonify({"text": ""})
        
        raw_text = data.get('text', '').strip()
        trigger_word = "@aibot"

        if raw_text.startswith(trigger_word):
            user_input = raw_text[len(trigger_word):].strip()
        else:
            user_input = raw_text

        if not user_input:
            return jsonify({"text": "Bạn chưa nhập nội dung câu hỏi."})

        answer_text = rag_service.get_answer(user_input)
        asker = data.get('user_name', 'bạn')
        return jsonify({
            "text":f"@{asker} {answer_text}" 
        })
    except Exception as e:
        return jsonify({
            "text": f"Hệ thống AI gặp lỗi: {str(e)}"
        })
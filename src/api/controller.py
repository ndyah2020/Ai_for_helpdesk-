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
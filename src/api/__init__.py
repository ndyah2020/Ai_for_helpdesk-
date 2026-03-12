from flask import Flask
from src.api.controller import chat_bp
from src.api.service import rag_service
from config.settings import settings
def create_app():
    app = Flask(__name__)

    # Enable CORS for all routes (allow calls from frontend apps on other origins)
    @app.after_request
    def _apply_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    with app.app_context():
        try:  
            rag_service.initialize()
        except Exception:
            print("Cảnh báo: Không thể load RAG Model lúc khởi động.")

    app.register_blueprint(chat_bp, url_prefix=settings.URL_PREFIX)

    return app
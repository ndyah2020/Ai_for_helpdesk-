from src.api import create_app
app = create_app()
# sử dụng postman với api http://localhost:5000/api/v1/chat key là input
if __name__ == "__main__":
    print("Server đang khởi động...")
    app.run(host="0.0.0.0", port=5000, debug=False)
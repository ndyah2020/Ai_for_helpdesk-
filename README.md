# AI Chính xác dùng cho tư vấn khách hàng
## 1. Mục đích
Xây dựng một hệ thống Hỏi-Đáp (RAG) có khả năng đọc và hiểu một tài liệu PDF để trả lời các câu hỏi của người dùng dựa trên nội dung tài liệu đó. Áp dụng vào việc tư vấn khách hàng của một nghiệp vụ nào đó.

## 2. Nguồn dữ liệu thử nghiệm
- Báo cáo Chỉ số Thương mại điện tử Việt Nam 2025: [Báo cáo EBI 2025](https://drive.google.com/file/d/18hUNrKSJXQmKOQl7mLhqhV1bg2MiKcmN/view)

## 3. Cấu trúc thư mục
RAG_PROJECT/                     <br>
├── config/                      <br> 
│   ├── setting.py               <br>
├── data/                        <br>
│   ├── persistent_chroma_db/    <br>
│   ├── source_documents/        <br>
├── src/                         <br>
│   ├── components/              <br>
│   ├── pipelines/               <br>
│   ├── api/                     <br>
│   │   ├── __init__.py          <br>
│   │   ├── controller.py        <br>
│   │   ├── service.py           <br>
│   │   └── schemas.py           <br>
│   ├── interfaces.py            <br>
├── unit_test/                   <br>
├── server.py                    <br>
├── run_ingestion.py             <br>
├── requirements.txt             <br>


- Trong đó
    + **source_documents** chứa file pdf là nguồn thông tin

## 4. Các bước chạy dự án với docker
### 4.1. Cài đặt docker
- Kiểm tra phiên bản docker hiện tại
  ```
  docker --version
  ```

- Nếu chưa có: [Download Docker](https://docs.docker.com/engine/install/)

### 4.2. Cài Ollama
- Tải Ollama: [Download Ollama](https://ollama.com/download)
- Kiểm tra:
  ```
  ollama --version
  ```

### 4.3. Tải các model cần thiết
- Tải model
  ```
  ollama pull bge-m3
  ```

- Kiểm tra các mô hình hiện có
  ```
  ollama list
  ```
### 4.4. Thiết lập môi trường để 
- Do Ollama mặc định bind 127.0.0.1 nên container trong docker sẽ không truy cập được nên phải set 
  ```
  OLLAMA_HOST=0.0.0.0
  ``` 

- Mục đích để Ollama lắng nghe trên mọi interface 
#### 4.4.1 Các bước thực hiện set ollama_host trên windows
- Truy cập vào **edit the system environment variables**

- Chọn **environment variable**

- Chọn **New** ở **System Variables**

- Nhập Valiable Name : ```OLLAMA_HOST```

- Nhập Valiable Value: ```0.0.0.0```
### 4.5 Chạy dự án
- Truy cập vào thư mục chính dự án và dùng lệnh
```
docker compose up -d --build
```

- Sau khi hoàn tất các thư viện đã được cài đặt dùng lệnh 
```
docker exec -it rag_app_server python run_ingestion.py
```

- Mục đích để nạp dữ liệu từ file pdf vào database
### 4.5 Xem kết quả 
- Dùng Postman 
- Chọn **body** -> **raw** với nội dung
```
{
    "input": "Xin Chào"
}
```

- Chọn phương thức **POST** gọi đến api 
```
http://localhost:5000/api/v1/chat
```

- Nếu chưa có: [Download Postman](https://www.postman.com/downloads/)

Chờ đợi và xem kết quả
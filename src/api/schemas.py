from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional, Literal

class ChatRequest(BaseModel):
    input: str = Field(..., description="Câu hỏi của người dùng", min_length=1)

    @field_validator('input')
    def validate_input(cls, v: str):
        v = v.strip()
        if not v:
            raise ValueError('Câu hỏi không được chỉ chứa khoảng trắng')
        return v
    
class AnswerData(BaseModel):
    input: str
    answer: str

class ChatResponse(BaseModel):
    status: Literal["success", "error"] = "success"
    data: AnswerData | None = None
    message: str | None = None
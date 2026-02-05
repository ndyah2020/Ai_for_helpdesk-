from abc import ABC, abstractmethod 
from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
# đọc file 
class BaseLoader(ABC):
    @abstractmethod
    def get_file_paths(self) -> list[str]:
        pass
    @abstractmethod
    def load_single_file(self, file_path: str) -> list[Document]:
        pass
    @abstractmethod
    def load_documents(self) -> list[Document]:
        pass

# chia nhỏ file thành các chunk
class BaseSplitter(ABC):
    @abstractmethod
    def split_documents(self, documents: list[Document]):
        pass

# lưu các chunk vào
class BaseVectorDB(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        pass
    @abstractmethod
    def reset_db(self):
        pass
    @abstractmethod
    def get_retriever(self, k: int):
        pass
    @abstractmethod
    def delete_file_data(seft, file_path:str):
        pass
    
#nhận vào các chunk và trả ra các chunk phù hợp với câu hỏi 
class BaseReranker(ABC):
    @abstractmethod
    def get_compressor(self):
        pass

# định dạng llm và prompt
class BaseLLM(ABC):
    @property
    @abstractmethod
    def llm(self) -> BaseLanguageModel:
        pass

    @property
    @abstractmethod
    def prompt(self) -> ChatPromptTemplate:
        pass

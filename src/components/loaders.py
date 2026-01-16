from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from src.interfaces import BaseLoader
import os
import glob

class LocalDirLoader(BaseLoader):
    def __init__(self, directory_path: str, glob_pattern: str = "*.pdf"):
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern

    def load_documents(self) -> List[Document]:
        search_path = os.path.join(self.directory_path, self.glob_pattern)
        pdf_paths = glob.glob(search_path)
        
        documents = []
        for path in pdf_paths:
            loader = PyMuPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
        return documents
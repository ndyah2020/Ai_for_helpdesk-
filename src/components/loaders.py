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

    def get_file_paths(self) -> List[str]:
        search_path = os.path.join(self.directory_path, self.glob_pattern)
        return glob.glob(search_path)

    def load_single_file(self, file_path: str) -> List[Document]:
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    def load_documents(self) -> List[Document]:
        file_paths = self.get_file_paths()
        documents = []
        for path in file_paths:
            documents.extend(self.load_single_file(path))
        return documents
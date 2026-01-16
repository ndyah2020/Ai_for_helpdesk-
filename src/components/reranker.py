from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from src.interfaces import BaseReranker

class HuggingFaceReranker(BaseReranker):
    def __init__(self, model_name: str , top_n: int):
        self.model = HuggingFaceCrossEncoder(model_name=model_name)
        self.compressor = CrossEncoderReranker(model=self.model, top_n=top_n)

    def get_compressor(self):
        return self.compressor
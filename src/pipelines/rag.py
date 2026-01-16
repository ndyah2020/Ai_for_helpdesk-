from src.interfaces import BaseVectorDB, BaseLLM
from config.settings import settings

class RAGpipeline:
    def __init__(self, vector_db: BaseVectorDB, llm_wrapper: BaseLLM):
        self.retriever = vector_db.get_retriever(k=settings.CHUNK_NUMBER)

        self.llm_client = llm_wrapper.llm
        self.prompt_template = llm_wrapper.prompt

    def _format_docs(self, docs):
        if not docs:
            return ""
        
        formatted_texts = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                formatted_texts.append(doc.page_content)
            
            elif isinstance(doc, dict):
                formatted_texts.append(doc.get('page_content', ''))
            
            elif isinstance(doc, tuple) and hasattr(doc[0], 'page_content'):
                formatted_texts.append(doc[0].page_content)
                
            else:
                formatted_texts.append(str(doc))
                
        return "\n\n".join(formatted_texts)
    
    def run(self, input: str) -> str: 
        retrieved_docs = self.retriever.invoke(input) 
        context_text = self._format_docs(retrieved_docs)
        
        final_prompt_content = self.prompt_template.format(
            context=context_text,
            input=input
        )

        messages = [
            {
                'role': 'user',
                'content': final_prompt_content
            },
        ]

        full_response = ""
        
        for part in self.llm_client.chat(model=settings.MODEL_NAME, messages=messages, stream=True):
            content = part['message']['content']
            # print(content, end='', flush=True)
            full_response += content
            
        return full_response
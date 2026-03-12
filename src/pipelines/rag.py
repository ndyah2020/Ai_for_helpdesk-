from src.interfaces import BaseVectorDB, BaseLLM, BaseReranker 
from config.settings import settings
from langchain.retrievers import ContextualCompressionRetriever

class RAGpipeline:
    def __init__(self, vector_db: BaseVectorDB, reranker: BaseReranker, llm_wrapper: BaseLLM):
        self.vector_db = vector_db
    
        self.compressor = reranker.get_compressor()


        self.llm_client = llm_wrapper.llm
        self.prompt_template = llm_wrapper.prompt
        self.outline_prompt_template = llm_wrapper.outline_prompt

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
    
    def run(self, input: str, k: int = None, mmr_diversity: float = 0.5) -> str: 
        # Xác định số lượng chunk muốn lấy (Dynamic K)
        target_k = k if k else settings.CHUNK_NUMBER

        try:
            print(f"Đang tìm kiếm với k={target_k}, diversity={mmr_diversity}...")
            #  TẠO BASE RETRIEVER ĐỘNG (Hybrid + MMR)
            base_retriever = self.vector_db.get_retriever(
                k=target_k, 
                mmr_diversity=mmr_diversity
            )
            #  GẮN RERANKER VÀO
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=base_retriever
            )
            # THỰC HIỆN TRUY VẤN
            retrieved_docs = compression_retriever.invoke(input)
            print(f"Tìm thấy {len(retrieved_docs)} tài liệu sau khi Rerank.")

        except Exception as e:
            print(f"Lỗi khi retrieve/rerank: {e}")
            retrieved_docs = []
        
        context_text = self._format_docs(retrieved_docs)
        
        print("Đang tạo dàn ý (Outline Generation)...")
        outline_prompt_content = self.outline_prompt_template.format(
            context=context_text,
            input=input
        )
        
        outline_messages = [{'role': 'user', 'content': outline_prompt_content}]
        outline_response = self.llm_client.chat(model=settings.MODEL_NAME, messages=outline_messages)
        outline_text = outline_response['message']['content']
        print(f"Dàn ý đã tạo:\n{outline_text}\n")

        print("Đang sinh các câu trả lời (Multi-trials)...")
        num_trials = 3 
        candidates = []
        
        enhanced_context = f"OUTLINE CẦN BÁM SÁT:\n{outline_text}\n\nTÀI LIỆU RAG NỀN TẢNG:\n{context_text}"
        
        final_prompt_content = self.prompt_template.format(
            context=enhanced_context,
            input=input
        )
        messages = [{'role': 'user', 'content': final_prompt_content}]

        for i in range(num_trials):
            print(f"  > Đang sinh ứng viên {i+1}...")
            response = self.llm_client.chat(
                model=settings.MODEL_NAME, 
                messages=messages,
                options={"temperature": 0.7 + i*0.1} 
            )
            candidate_html = response['message']['content']
            candidates.append(candidate_html)
    
        print("Đang đánh giá và chọn câu trả lời tốt nhất...")
        best_candidate = candidates[0] # Fallback
        
        eval_prompt = f"""
        Given the original question, the generated outline, and {num_trials} generated HTML responses, choose the index (1, 2, or 3) of the response that:
        1. Best follows the outline.
        2. Has the most valid and professional HTML structure.
        3. Answers the user question completely and empathetically in Vietnamese.
        
        Question: {input}
        Outline: {outline_text}
        
        Candidates:
        """
        for idx, cand in enumerate(candidates):
            eval_prompt += f"\n--- Candidate {idx+1} ---\n{cand}\n"
            
        eval_prompt += "\nOutput ONLY the number of the best candidate (e.g., '1', '2', or '3'). No other text."
        
        try:
            eval_response = self.llm_client.chat(
                model=settings.MODEL_NAME,
                messages=[{'role': 'user', 'content': eval_prompt}],
                options={"temperature": 0.0}
            )
            parsed_idx = int(eval_response['message']['content'].strip()) - 1
            if 0 <= parsed_idx < num_trials:
                best_candidate = candidates[parsed_idx]
                print(f"Đã chọn ứng viên số {parsed_idx + 1} làm câu trả lời tốt nhất.")
            else:
                 print("Không thể parse index, dùng ứng viên 1 mặc định.")
        except Exception as e:
            print(f"Lỗi khi đánh giá ({e}), dùng ứng viên 1 mặc định.")
 
        return best_candidate
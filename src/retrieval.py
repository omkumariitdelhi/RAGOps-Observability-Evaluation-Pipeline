import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from sentence_transformers import CrossEncoder
from langfuse import observe
from src.config import DB_DIR, EMBEDDING_MODEL_NAME, CROSS_ENCODER_MODEL_NAME

logger = logging.getLogger(__name__)

class HybridReRankedRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        logger.info(f"Loading ChromaDB from {DB_DIR}")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.db = Chroma(persist_directory=str(DB_DIR), embedding_function=self.embeddings)
        
        # Load cross-encoder
        logger.info(f"Loading Cross-Encoder {CROSS_ENCODER_MODEL_NAME}")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        
        # Create Vector Retriever
        self.vector_retriever = self.db.as_retriever(search_kwargs={"k": top_k * 2})
        
        self._init_bm25_retriever()

    def _init_bm25_retriever(self):
        try:
            db_data = self.db.get()
            docs = [
                Document(page_content=content, metadata=meta) 
                for content, meta in zip(db_data['documents'], db_data['metadatas'])
            ]
            if docs:
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                self.bm25_retriever.k = self.top_k * 2
            else:
                self.bm25_retriever = None
        except Exception as e:
            logger.warning(f"Could not initialize BM25 Retriever: {e}")
            self.bm25_retriever = None

    @observe()
    def retrieve(self, query: str) -> List[Document]:
        logger.info(f"Retrieving candidate chunks for query: '{query}'")
        candidates = []
        
        # 1. Semantic Search
        vector_docs = self.vector_retriever.invoke(query)
        candidates.extend(vector_docs)
        
        # 2. Keyword Search (BM25)
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            candidates.extend(bm25_docs)
            
        if not candidates:
            return []
            
        # Deduplicate candidates by page_content
        unique_candidates = []
        seen = set()
        for doc in candidates:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_candidates.append(doc)

        # 3. Re-rank chunks using Cross-Encoder
        pairs = [[query, doc.page_content] for doc in unique_candidates]
        scores = self.cross_encoder.predict(pairs)
        
        # 4. Sort candidates by score descending
        scored_candidates = list(zip(unique_candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Return top_k
        top_docs = [doc for doc, score in scored_candidates[:self.top_k]]
        return top_docs

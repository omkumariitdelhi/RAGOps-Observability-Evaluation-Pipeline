import logging
import time
import httpx
from langfuse import observe, Langfuse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.retrieval import HybridReRankedRetriever
from src.config import PROMPTS, ACTIVE_MODELS, OPENROUTER_API_KEY, MODEL_API_KEYS, OPENAI_API_BASE, OLLAMA_API_BASE, OLLAMA_API_KEY, USE_OLLAMA

logger = logging.getLogger(__name__)

# Module-level singleton — avoid re-loading 100MB+ models per request
_shared_retriever = None

def get_shared_retriever(top_k: int = 5) -> HybridReRankedRetriever:
    """Return a singleton retriever so embedding + cross-encoder models
    are loaded once and reused across all requests."""
    global _shared_retriever
    if _shared_retriever is None:
        logger.info("Initializing shared HybridReRankedRetriever (one-time)...")
        _shared_retriever = HybridReRankedRetriever(top_k=top_k)
    return _shared_retriever


def _make_http_clients(timeout: float = 120.0):
    """Create httpx clients that won't trip over Windows-invalid characters
    (colons in OpenRouter model IDs like 'llama-3.3-70b-instruct:free').

    By constructing our own httpx.Client we avoid any internal caching /
    temp-file logic that would embed the model name in a file path.
    """
    transport = httpx.HTTPTransport(retries=2)
    async_transport = httpx.AsyncHTTPTransport(retries=2)

    client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(timeout),
    )
    async_client = httpx.AsyncClient(
        transport=async_transport,
        timeout=httpx.Timeout(timeout),
    )
    return client, async_client


class RAGPipeline:
    def __init__(self, model_name=None, retriever=None):
        # Re-use a shared retriever if provided, otherwise get the singleton
        self.retriever = retriever if retriever else get_shared_retriever(top_k=5)
        selected_model = model_name if model_name else ACTIVE_MODELS[0]

        if USE_OLLAMA:
            logger.info(f"Using local Ollama with model: {selected_model}")
            sync_client, async_client = _make_http_clients()
            self.llm = ChatOpenAI(
                model=selected_model,
                temperature=0,
                api_key=OLLAMA_API_KEY,
                base_url=OLLAMA_API_BASE,
                http_client=sync_client,
                http_async_client=async_client,
            )
        else:
            api_key = MODEL_API_KEYS.get(selected_model, OPENROUTER_API_KEY)
            if not api_key:
                logger.warning("OPENROUTER_API_KEY is not set. Generation might fail.")
            logger.info(f"Using OpenRouter with model: {selected_model}")
            sync_client, async_client = _make_http_clients()
            self.llm = ChatOpenAI(
                model=selected_model,
                temperature=0,
                api_key=api_key,
                base_url=OPENAI_API_BASE,
                http_client=sync_client,
                http_async_client=async_client,
                max_retries=5,
            )

        system_msg = SystemMessagePromptTemplate.from_template(PROMPTS['generation']['system'])
        human_msg = HumanMessagePromptTemplate.from_template(PROMPTS['generation']['user'])
        self.prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

    def format_docs(self, docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Chunk [{i+1}] (Source: {source}):\n{doc.page_content}")
        return "\n\n".join(formatted)

    @observe()
    def generate_answer(self, question: str):
        lf = Langfuse()
        docs = self.retriever.retrieve(question)
        if not docs:
            return {
                "answer": "I'm sorry, but I cannot answer this question based on the provided evidence.",
                "context": [],
                "trace_id": lf.get_current_trace_id()
            }

        context_str = self.format_docs(docs)
        chain = self.prompt | self.llm
        
        response = chain.invoke({
            "context": context_str,
            "question": question
        })

        return {
            "answer": response.content,
            "context": [doc.page_content for doc in docs],
            "trace_id": lf.get_current_trace_id()
        }

    async def generate_answer_from_context_async(self, question: str, docs):
        """Asynchronously generate answer using pre-retrieved docs."""
        if not docs:
            return "I'm sorry, but I cannot answer this question based on the provided evidence."
            
        context_str = self.format_docs(docs)
        chain = self.prompt | self.llm
        
        response = await chain.ainvoke({
            "context": context_str,
            "question": question
        })
        
        return response.content

import logging
import time
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.generation import RAGPipeline, get_shared_retriever
from src.config import ACTIVE_MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    results: dict
    contexts: list

# Pre-warm the retriever at startup so the first request isn't slow
@app.on_event("startup")
def warmup():
    logger.info("Pre-loading retriever models...")
    get_shared_retriever(top_k=5)
    logger.info("Retriever ready.")

@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    logger.info(f"Received query: {request.query}")

    # Get the shared retriever once
    retriever = get_shared_retriever(top_k=5)

    # 1. Retrieve the relevant documents ONCE
    # Since the retriever might have blocking IO (like Chroma), run it in a thread
    docs = await asyncio.to_thread(retriever.retrieve, request.query)
    contexts = [doc.page_content for doc in docs] if docs else []

    # 2. Define the async task for a single model
    async def query_model(model):
        logger.info(f"Querying model: {model}")
        try:
            pipeline = RAGPipeline(model_name=model, retriever=retriever)
            answer = await pipeline.generate_answer_from_context_async(request.query, docs)
            return model, answer
        except Exception as e:
            logger.error(f"Error querying {model}: {e}")
            return model, f"Error generating response: {str(e)}"

    # 3. Query all ACTIVE_MODELS in parallel
    tasks = [query_model(model) for model in ACTIVE_MODELS]
    completed_tasks = await asyncio.gather(*tasks)

    # Reconstruct the results dict
    results = {model: answer for model, answer in completed_tasks}

    return QueryResponse(
        query=request.query,
        results=results,
        contexts=contexts
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

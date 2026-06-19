# RAGOps Observability Evaluation Pipeline

An advanced, production-ready Retrieval-Augmented Generation (RAG) application with a modern frontend and a scalable FastAPI backend. This pipeline allows for parallel querying of multiple state-of-the-art Large Language Models (LLMs) to compare responses, ensuring robustness and high-quality generation.

## 🚀 Features

- **Parallel LLM Querying:** Simultaneously query multiple OpenRouter models (e.g., LLaMA 3.3, Kimi K2.6, Gemma, OpenAI models) in parallel using `asyncio` for rapid inference and response comparison.
- **Advanced Retrieval:** Employs a pre-warmed singleton `HybridReRankedRetriever` combining dense vector search (ChromaDB) with cross-encoder reranking (`ms-marco-MiniLM`) for highly accurate context fetching.
- **Sleek Frontend:** A beautifully designed glassmorphic UI built with React. Features a dropdown selector to cleanly toggle between different model responses in real-time.
- **Extensible Architecture:** Designed with maintainability in mind. Easy to swap embedding models, generation endpoints, and observability hooks.

## 📂 Project Structure

```
.
├── Docs/                  # PDF documents for vector database ingestion
├── config/                # YAML configuration for prompts and pipelines
├── chroma_db/             # Local vector database (persisted)
├── frontend/              # React frontend application
│   ├── src/               # UI components, styles (index.css), and API bindings
│   └── package.json       # Node dependencies
├── src/                   # FastAPI backend application
│   ├── api.py             # FastAPI entrypoint and endpoints
│   ├── generation.py      # LLM wrappers and RAG pipeline logic
│   ├── retrieval.py       # Embedding models and reranking logic
│   └── config.py          # Environment variables and API keys mapping
├── tests/                 # Quality and regression testing
└── requirements.txt       # Python backend dependencies
```

## 🛠️ Setup Instructions

### 1. Backend Setup

1. **Create a Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration:**
   - Create a `.env` file in the root directory.
   - Add your API keys securely:
     ```env
     OPENROUTER_API_KEY=your_key_here
     ```
   *(Note: Specific OpenRouter models may have custom keys assigned in `src/config.py`)*

4. **Run the Backend:**
   ```bash
   python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

### 2. Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node modules:**
   ```bash
   npm install
   ```

3. **Run the Development Server:**
   ```bash
   npm run dev
   ```

## 🧠 Usage

1. Open your browser and navigate to the local frontend server (typically `http://localhost:5173`).
2. Type a query into the sleek search bar.
3. The backend will simultaneously execute a high-accuracy retrieval and ping multiple configured models.
4. Use the dropdown menu in the UI to seamlessly flip between how different LLMs answered your query based on the retrieved context.

## 🛡️ License

This project is licensed under the MIT License.

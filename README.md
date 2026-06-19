# RAGOps Observability & Evaluation Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Langfuse](https://img.shields.io/badge/Langfuse-Observability-orange)
![Ragas](https://img.shields.io/badge/Ragas-LLM%20Evaluation-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-SRE%20Dashboard-red)
![OpenRouter](https://img.shields.io/badge/OpenRouter-Multi--LLM-black)

A production-grade Retrieval-Augmented Generation (RAG) evaluation and observability pipeline. This project systematically tests multiple leading open-source LLMs against a custom knowledge base, evaluates their faithfulness/hallucination rates using an LLM-as-a-Judge (GPT-4o), traces the execution using **Langfuse**, and visualizes real-time SRE metrics via a **Streamlit Dashboard**.

---

## 🏗️ Architecture & Features

This pipeline mimics a true MLOps/RAGOps workflow:

1. **Multi-Model Regression Testing (`tests/test_rag_regression.py`)**
   - Iterates through multiple LLMs (e.g., Gemma-4, GPT-OSS, Kimi).
   - Generates answers across a curated set of benchmark questions (`rag_evaluation_questions.md`).
2. **LLM-as-a-Judge Evaluation (`ragas`)**
   - Employs **Ragas** (specifically the `faithfulness` metric) leveraging GPT-4o to evaluate the baseline generations for hallucinations.
3. **Observability & Tracing (`Langfuse`)**
   - Uses the Langfuse SDK (`@observe()`) to seamlessly intercept prompts, retrieval contexts, and generation latencies.
   - Pushes execution traces and Ragas scores directly to the Langfuse Cloud.
4. **SRE Metrics Dashboard (`src/dashboard/app.py`)**
   - A custom Streamlit dashboard that continuously pulls scoring telemetry from the Langfuse API.
   - Calculates a true moving average of generation quality across all deployed LLMs.
5. **CI/CD Integration**
   - Includes a GitHub Actions workflow (`.github/workflows/rag_eval.yml`) to run the regression suite automatically on pull requests.

---

## 📂 Repository Structure

```text
.
├── src/
│   ├── dashboard/       # Streamlit SRE Dashboard
│   │   └── app.py       # Fetches telemetry from Langfuse API
│   └── rag_app/         # Core Generation & Prompt logic
│       ├── pipeline.py  # Hybrid retriever & generator wrapper
│       └── prompts.py   # RAG System Prompts
├── tests/
│   ├── test_rag_regression.py  # Pytest suite for multi-model Ragas evaluation
│   └── golden_dataset.json     # Ground truth for baseline comparisons
├── parse_md.py                 # Utility to ingest and parse custom markdown datasets
├── rag_evaluation_questions.md # Benchmark evaluation questions
├── .env.example                # Template for required API keys
└── requirements.txt            # Python dependencies
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/omkumariitdelhi/RAGOps-Observability-Evaluation-Pipeline.git
cd RAGOps-Observability-Evaluation-Pipeline
python -m venv .venv
source .venv/Scripts/activate  # On Windows
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Populate it with your credentials:
- **Langfuse:** Get your keys from [Langfuse Cloud](https://cloud.langfuse.com)
- **OpenRouter:** Get your multi-model access keys from [OpenRouter](https://openrouter.ai) (Ensure you have credits for the evaluator model).

```env
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"

OPENROUTER_API_KEY="sk-or-..."
GPT4O_API_KEY="sk-or-..."
```

*(Note: Never commit your `.env` file. It is safely ignored in `.gitignore`.)*

### 3. Run the Evaluation Suite

Kick off the Pytest suite to generate answers across all models and score them with GPT-4o:

```bash
pytest tests/test_rag_regression.py -s -v
```

As the test runs, traces and `faithfulness` scores will immediately populate in your Langfuse Cloud project.

### 4. Launch the SRE Dashboard

In a separate terminal window, launch the Streamlit dashboard to monitor the moving average of your RAG metrics:

```bash
streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) to view the live dashboard.

---

## 🛠️ Tech Stack

* **Frameworks:** LangChain, Streamlit, Pytest
* **Observability:** Langfuse SDK & API
* **Evaluation:** Ragas (Retrieval Augmented Generation Assessment)
* **LLMs:** OpenAI (GPT-4o), Anthropic, Meta (Llama-3), Google (Gemma) via OpenRouter

---
*Built to showcase production-grade AI observability and MLOps engineering capabilities.*

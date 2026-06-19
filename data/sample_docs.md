# Architecture Overview
The Enterprise Knowledge Assistant uses a hybrid retrieval pipeline. It combines dense vector semantic search using HuggingFace all-MiniLM-L6-v2 embeddings stored in ChromaDB, with sparse keyword retrieval using BM25. 
To improve precision, retrieved candidates from both methods are re-ranked using a cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2.

# Data Ingestion
Documents are split into manageable chunks for vectorization. The system uses a standard chunk size of 500 tokens with an overlap of 100 tokens. This overlap is crucial to ensure that important context spanning chunk boundaries is preserved.

# Continuous Evaluation
The system prevents regression in answer quality by integrating Ragas into the CI/CD pipeline. Every PR triggers an evaluation against a golden dataset. If the faithfulness score drops below 0.85, the build is automatically rejected. This gating prevents hallucination-prone prompts or models from reaching production.

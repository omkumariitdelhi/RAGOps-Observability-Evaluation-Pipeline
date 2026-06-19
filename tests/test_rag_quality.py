import json
import pytest
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from src.config import DATA_DIR
import logging

logger = logging.getLogger(__name__)

def load_golden_dataset():
    path = DATA_DIR / "golden_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_rag_faithfulness():
    data = load_golden_dataset()
    
    dataset_dict = {
        "question": [row["question"] for row in data],
        "answer": [row["answer"] for row in data],
        "contexts": [row["contexts"] for row in data],
        "ground_truth": [row["ground_truth"] for row in data],
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    logger.info("Starting Ragas evaluation...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    
    print("Ragas Evaluation Results:", result)
    
    # CI/CD Gating: Fail if faithfulness drops below 0.85
    faithfulness_score = result.get("faithfulness", 0)
    assert faithfulness_score >= 0.85, f"Faithfulness ({faithfulness_score}) is below threshold 0.85"

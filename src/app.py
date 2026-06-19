import argparse
from src.generation import RAGPipeline
from src.config import FREE_MODELS
import logging

logging.basicConfig(level=logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG CLI")
    parser.add_argument("--query", type=str, required=True, help="Question to ask the RAG system")
    args = parser.parse_args()

    print("\n==================================")
    print(f"Question: {args.query}")
    print("==================================\n")

    first_result = None

    for model in FREE_MODELS:
        print(f"\n--- Model: {model} ---")
        try:
            pipeline = RAGPipeline(model_name=model)
            result = pipeline.generate_answer(args.query)
            print("\nAnswer:")
            print(result['answer'])
            if first_result is None:
                first_result = result
        except Exception as e:
            print(f"Error with model {model}: {e}")

    # Print contexts only once (from the first successful model)
    if first_result and first_result['context']:
        print("\n==================================")
        print("Retrieved Contexts (used by all models)")
        print("==================================")
        for i, ctx in enumerate(first_result['context']):
            print(f"\n[Chunk {i+1}]: {ctx}")

if __name__ == "__main__":
    main()

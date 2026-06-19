import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "Docs"
DB_DIR = BASE_DIR / "chroma_db"

def load_prompts():
    prompt_path = CONFIG_DIR / "prompts.yaml"
    if not prompt_path.exists():
        return {}
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
# Swapped to the new key provided by the user to bypass the 50/day free limit on the previous key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

# Default settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# Define embedding models
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# In a real setup, load these from environment variables securely
MODEL_API_KEYS = {
    "openai/gpt-4o": os.getenv("GPT4O_API_KEY", ""),
    "anthropic/claude-3.5-sonnet": os.getenv("CLAUDE_API_KEY", "")
}

# Free models from OpenRouter (generators to evaluate)
FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "moonshotai/kimi-k2.6:free"
]

OLLAMA_MODELS = [
    "llama3",
    "mistral"
]

ACTIVE_MODELS = OLLAMA_MODELS if USE_OLLAMA else FREE_MODELS

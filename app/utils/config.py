import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", BASE_DIR / "data/raw"))
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", BASE_DIR / "data/vectordb"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
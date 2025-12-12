"""
Centralized configuration for the Graph RAG Research Assistant.
All settings are loaded from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GRAVIX_API_KEY = os.environ.get("GRAVIX_API_KEY")

# --- LLM Settings ---
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

# --- Embedding Settings ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text:v1.5")
EMBEDDING_DIMENSION = 768

# --- Graph Settings ---
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.5"))
EDGE_DISPLAY_THRESHOLD = float(os.environ.get("EDGE_DISPLAY_THRESHOLD", "0.35"))

# --- Search Settings ---
DEFAULT_MAX_RESULTS = int(os.environ.get("DEFAULT_MAX_RESULTS", "15"))
MAX_RESULTS_LIMIT = 50  # Hard limit to prevent abuse

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss")

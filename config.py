"""Central configuration for ScopeGuard.

One place for paths, model names, and knobs so the rest of the code
never hardcodes them. Tweak here, not in twelve files.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
KB_CHUNKS_PATH = DATA_DIR / "kb_chunks.jsonl"      # output of build_kb.py
FAISS_INDEX_PATH = DATA_DIR / "kb.index"           # output of rag/index.py
CHUNK_META_PATH = DATA_DIR / "kb_meta.jsonl"       # chunk text aligned to index
QUERIES_PATH = DATA_DIR / "queries.jsonl"          # labeled eval queries
CLASSIFIER_DIR = ROOT / "classifier" / "model"     # saved DistilBERT fine-tune
RESULTS_PATH = ROOT / "eval" / "results.csv"       # output of run_eval.py
PARETO_PLOT_PATH = ROOT / "eval" / "pareto.png"    # output of metrics.py

# --- Models ------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"
CLASSIFIER_BASE = "distilbert-base-uncased"

# --- Retrieval ---------------------------------------------------------
TOP_K = 4              # chunks retrieved per query
CHUNK_SIZE = 800       # characters per KB chunk
CHUNK_OVERLAP = 100    # character overlap between chunks

# --- Classifier training (CPU-friendly defaults) -----------------------
# No GPU: small batch, few epochs. ~300 short queries trains in minutes.
TRAIN_EPOCHS = 4
TRAIN_BATCH_SIZE = 8
TRAIN_TEST_SPLIT = 0.2   # fraction held out for testing
SCOPE_THRESHOLD = 0.5    # P(out-of-scope) above this -> deflect

# --- API key -----------------------------------------------------------
# Loaded from .env (see .env.example). Never hardcode the key.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# --- Canned deflection response ----------------------------------------
DEFLECTION_MESSAGE = (
    "I can only answer questions about the products and policies in my "
    "knowledge base. For personalized investment, tax, or legal advice, "
    "please consult a licensed professional."
)

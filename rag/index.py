"""FAISS index over the knowledge base using text-embedding-3-small.

build_index() embeds every KB chunk and saves a FAISS index plus an
aligned metadata file. retrieve() embeds a query and returns the
top-k most similar chunks.

Run build_index() once after data/build_kb.py, before any evaluation.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))
import config

_client = None


def _get_client():
    """Lazy OpenAI client so importing this module never needs a key."""
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env.")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def embed_texts(texts):
    """Embed a list of strings. Returns a float32 numpy array (n, dim)."""
    resp = _get_client().embeddings.create(
        model=config.EMBED_MODEL, input=texts
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")


def build_index():
    """Embed all KB chunks and save the FAISS index + metadata."""
    if not config.KB_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{config.KB_CHUNKS_PATH} missing. Run data/build_kb.py first."
        )

    chunks = [json.loads(line) for line in open(config.KB_CHUNKS_PATH)]
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {config.EMBED_MODEL}...")

    vectors = embed_texts(texts)
    faiss.normalize_L2(vectors)  # cosine similarity via inner product

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    # Save chunk metadata in the same order as vectors were added.
    with open(config.CHUNK_META_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Index built: {index.ntotal} vectors -> {config.FAISS_INDEX_PATH}")


# Loaded once and reused across retrieve() calls.
_index = None
_meta = None


def _load_index():
    global _index, _meta
    if _index is None:
        if not config.FAISS_INDEX_PATH.exists():
            raise FileNotFoundError("Index missing. Run build_index() first.")
        _index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        _meta = [json.loads(line) for line in open(config.CHUNK_META_PATH)]
    return _index, _meta


def retrieve(query, k=config.TOP_K):
    """Return the top-k KB chunks for a query as a list of dicts.

    Each dict: {id, source, text, score}. Higher score = more similar.
    """
    index, meta = _load_index()
    qvec = embed_texts([query])
    faiss.normalize_L2(qvec)
    scores, idxs = index.search(qvec, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:  # FAISS pads with -1 if fewer than k results
            continue
        chunk = dict(meta[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


if __name__ == "__main__":
    build_index()

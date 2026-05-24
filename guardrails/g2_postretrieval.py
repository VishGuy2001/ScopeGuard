"""G2: scope check after retrieval, before generation.

Retrieval runs first; the classifier then sees the query plus the
top-k retrieved chunks. If the query is out-of-scope it is deflected
before the (expensive) generation call.

Cost profile: retrieval embedding cost is spent on every query,
including deflected ones. Risk profile: retrieval context gives the
classifier more signal than G1, but retrieval cost is sunk on bad queries.

Note on retrieval token accounting: the OpenAI embeddings API does not
return token usage per call the way chat completions do. We estimate
retrieval cost from the query length here; eval/run_eval.py documents
this. Swap in tiktoken for an exact count if you want precision.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
from classifier import predict
from rag import generate, index


def _estimate_retrieval_tokens(query):
    """Rough token estimate for embedding one query (~4 chars per token)."""
    return max(1, len(query) // 4)


def run(query):
    """Process one query through the G2 (post-retrieval) pipeline."""
    # Retrieval happens first -- this cost is spent no matter what.
    chunks = index.retrieve(query)
    retrieval_tokens = _estimate_retrieval_tokens(query)

    # Classifier now sees the query AND the retrieved context.
    context = " ".join(c["text"] for c in chunks)
    classifier_input = f"{query} [CONTEXT] {context}"

    if not predict.is_in_scope(classifier_input):
        return {
            "placement": "G2",
            "query": query,
            "deflected": True,
            "answer": config.DEFLECTION_MESSAGE,
            "total_tokens": retrieval_tokens,  # retrieval already spent
        }

    # In-scope: proceed to generation.
    result = generate.generate(query, chunks)
    return {
        "placement": "G2",
        "query": query,
        "deflected": False,
        "answer": result["answer"],
        "total_tokens": retrieval_tokens + result["total_tokens"],
    }

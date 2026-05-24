"""G3: scope check on the generated response, before returning it.

The full pipeline runs -- retrieval and generation both complete --
and the classifier then inspects the generated answer for prohibited
advice language. If flagged, the answer is replaced with the canned
deflection response.

Cost profile: full retrieval + generation cost on EVERY query,
including out-of-scope ones. Risk profile: the classifier judges an
actual answer, the strongest signal, so false positives are lowest --
but maximum tokens are spent on out-of-scope traffic.
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
    """Process one query through the G3 (post-generation) pipeline."""
    # Full pipeline runs first -- all cost is spent before the check.
    chunks = index.retrieve(query)
    retrieval_tokens = _estimate_retrieval_tokens(query)
    result = generate.generate(query, chunks)
    total_tokens = retrieval_tokens + result["total_tokens"]

    # Classifier inspects the generated answer itself.
    if not predict.is_in_scope(result["answer"]):
        return {
            "placement": "G3",
            "query": query,
            "deflected": True,
            "answer": config.DEFLECTION_MESSAGE,
            "total_tokens": total_tokens,  # everything was already spent
        }

    return {
        "placement": "G3",
        "query": query,
        "deflected": False,
        "answer": result["answer"],
        "total_tokens": total_tokens,
    }

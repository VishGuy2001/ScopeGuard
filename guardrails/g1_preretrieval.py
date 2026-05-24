"""G1: scope check on the raw query, before any retrieval or LLM call.

The classifier runs on query text alone. Out-of-scope queries are
deflected immediately with the canned response.

Cost profile: near-zero tokens on deflected queries (no embedding,
no generation). Risk profile: no retrieval context, so the classifier
must decide from the query wording alone -- weakest signal of the three.

Every guardrail's run() returns the same dict shape so eval/run_eval.py
can compare placements fairly:
    {placement, query, deflected, answer, total_tokens}
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
from classifier import predict
from rag import generate, index


def run(query):
    """Process one query through the G1 (pre-retrieval) pipeline."""
    # Scope check happens first, on raw text, before spending anything.
    if not predict.is_in_scope(query):
        return {
            "placement": "G1",
            "query": query,
            "deflected": True,
            "answer": config.DEFLECTION_MESSAGE,
            "total_tokens": 0,  # nothing was spent
        }

    # In-scope: run the full RAG pipeline.
    chunks = index.retrieve(query)
    result = generate.generate(query, chunks)
    return {
        "placement": "G1",
        "query": query,
        "deflected": False,
        "answer": result["answer"],
        "total_tokens": result["total_tokens"],
    }

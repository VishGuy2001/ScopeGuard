"""Answer generation with GPT-4o-mini, grounded on retrieved chunks.

generate() builds a grounded prompt from a query and its retrieved
KB chunks, calls GPT-4o-mini, and returns the answer together with
the token counts. Token counts feed the cost side of the cost-risk
analysis, so every call reports them.
"""

import sys
from pathlib import Path

from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))
import config

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env.")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


SYSTEM_PROMPT = (
    "You are a finance information assistant. Answer ONLY using the "
    "provided knowledge base context. If the context does not contain "
    "the answer, say you do not have that information. Do not give "
    "personalized investment, tax, or legal advice."
)


def generate(query, chunks):
    """Generate an answer grounded on retrieved chunks.

    Args:
        query: the user question.
        chunks: list of chunk dicts from rag.index.retrieve().

    Returns:
        dict with keys: answer, prompt_tokens, completion_tokens, total_tokens.
    """
    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    user_prompt = (
        f"Knowledge base context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above."
    )

    resp = _get_client().chat.completions.create(
        model=config.GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    usage = resp.usage
    return {
        "answer": resp.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }

"""Inference wrapper for the fine-tuned scope classifier.

Loads the DistilBERT model saved by train.py and exposes a simple
predict function. All three guardrails call this; they differ only
in WHAT text they pass in and WHEN.

The model is loaded once and cached on first use.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
import config

_model = None
_tokenizer = None


def _load():
    """Lazy-load the fine-tuned model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        if not config.CLASSIFIER_DIR.exists():
            raise FileNotFoundError(
                f"No model at {config.CLASSIFIER_DIR}. Run classifier/train.py first."
            )
        _tokenizer = AutoTokenizer.from_pretrained(str(config.CLASSIFIER_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(
            str(config.CLASSIFIER_DIR)
        )
        _model.eval()
    return _model, _tokenizer


def predict_proba(text):
    """Return P(out-of-scope) for a piece of text, a float in [0, 1]."""
    model, tokenizer = _load()
    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[1])  # index 1 = out_of_scope


def is_in_scope(text, threshold=config.SCOPE_THRESHOLD):
    """Return True if text is in-scope, False if out-of-scope.

    A query is deflected when P(out-of-scope) >= threshold.
    """
    return predict_proba(text) < threshold

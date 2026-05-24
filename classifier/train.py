"""Fine-tune DistilBERT as the in-scope / out-of-scope scope classifier.

Trains a binary classifier (0 = in_scope, 1 = out_of_scope) on the
labeled queries in data/queries.jsonl. The same fine-tuned model
powers all three guardrail placements (G1, G2, G3).

No GPU required: defaults in config.py keep batch size and epochs
small. ~300 short queries train in a few minutes on CPU.

Usage:
    python classifier/train.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.append(str(Path(__file__).parent.parent))
import config

LABEL2ID = {"in_scope": 0, "out_of_scope": 1}


def load_dataset():
    """Load queries.jsonl and split into train/test Hugging Face Datasets."""
    rows = [json.loads(line) for line in open(config.QUERIES_PATH)]
    texts = [r["query"] for r in rows]
    labels = [LABEL2ID[r["label"]] for r in rows]

    ds = Dataset.from_dict({"text": texts, "label": labels})
    ds = ds.train_test_split(
        test_size=config.TRAIN_TEST_SPLIT, seed=42, stratify_by_column="label"
    )
    return ds


def compute_metrics(eval_pred):
    """Accuracy + precision/recall/F1 for the Trainer's eval loop."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train():
    """Fine-tune DistilBERT and save the model to classifier/model/."""
    ds = load_dataset()
    print(f"Train: {len(ds['train'])}  Test: {len(ds['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_BASE)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=128)

    ds = ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_BASE, num_labels=2
    )

    args = TrainingArguments(
        output_dir=str(config.CLASSIFIER_DIR / "checkpoints"),
        num_train_epochs=config.TRAIN_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAIN_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",  # watch curves: tensorboard --logdir
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nFinal test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    trainer.save_model(str(config.CLASSIFIER_DIR))
    tokenizer.save_pretrained(str(config.CLASSIFIER_DIR))
    print(f"\nModel saved -> {config.CLASSIFIER_DIR}")


if __name__ == "__main__":
    train()

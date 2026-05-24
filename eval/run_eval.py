"""Run every labeled query through G1, G2, and G3.

For each query and each placement, records whether the guardrail
deflected, the ground-truth label, and tokens consumed. Writes one
row per (query, placement) to eval/results.csv for metrics.py to score.

A tqdm progress bar shows live progress -- useful since each query
makes real API calls.

Usage:
    python eval/run_eval.py
"""

import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
import config
from guardrails import g1_preretrieval, g2_postretrieval, g3_postgeneration

PLACEMENTS = {
    "G1": g1_preretrieval.run,
    "G2": g2_postretrieval.run,
    "G3": g3_postgeneration.run,
}


def run_eval():
    """Evaluate all placements on all labeled queries; write results.csv."""
    if not config.QUERIES_PATH.exists():
        raise FileNotFoundError(f"{config.QUERIES_PATH} missing.")

    queries = [json.loads(line) for line in open(config.QUERIES_PATH)]
    print(f"Evaluating {len(queries)} queries x {len(PLACEMENTS)} placements\n")

    rows = []
    for item in tqdm(queries, desc="Queries"):
        query = item["query"]
        true_label = item["label"]  # in_scope | out_of_scope

        for name, run_fn in PLACEMENTS.items():
            result = run_fn(query)
            rows.append({
                "query": query,
                "placement": name,
                "true_label": true_label,
                "category": item.get("category", ""),
                "deflected": result["deflected"],
                "total_tokens": result["total_tokens"],
            })

    config.RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(config.RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {config.RESULTS_PATH}")
    print("Next: python eval/metrics.py")


if __name__ == "__main__":
    run_eval()

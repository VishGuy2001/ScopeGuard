# ScopeGuard

Where should a scope guardrail live in a RAG pipeline? ScopeGuard benchmarks
pre-retrieval, post-retrieval, and post-generation guardrail placement on cost
and legal risk — with a reproducible finance-domain query benchmark.

## What this is

Domain-locked RAG chatbots (finance, insurance, legal) are only permitted to
answer what their knowledge base covers. A *scope guardrail* blocks
out-of-scope queries — "should I buy this stock?", "do I have grounds to sue?"
— before they become a liability.

A guardrail can sit at three points in the pipeline. ScopeGuard implements all
three on the same pipeline and measures the cost-vs-risk tradeoff of each:

| Placement | When the scope check fires | Cost | Risk |
|-----------|---------------------------|------|------|
| **G1** Pre-retrieval | On the raw query | Near-zero tokens | No context → weakest signal |
| **G2** Post-retrieval | On query + retrieved chunks | Retrieval tokens spent | More signal, retrieval cost sunk |
| **G3** Post-generation | On the generated answer | Full pipeline cost | Strongest signal, max spend |

## Pipeline

```
            G1                  G2                       G3
query ──▶ [scope?] ──▶ retrieve ──▶ [scope?] ──▶ generate ──▶ [scope?] ──▶ answer
```

The same fine-tuned DistilBERT classifier powers all three placements; they
differ only in *when* it fires and *what cost* is already spent.

## Repo structure

```
scopeguard/
├── config.py              # all paths, model names, knobs — edit here
├── data/
│   ├── build_kb.py         # scrape SEC FAQ pages → kb_chunks.jsonl
│   └── queries.jsonl       # labeled eval queries (committed; expand to 300)
├── rag/
│   ├── index.py            # FAISS index + text-embedding-3-small
│   └── generate.py         # GPT-4o-mini generation, with token tracking
├── classifier/
│   ├── train.py            # fine-tune DistilBERT (CPU-friendly)
│   └── predict.py          # inference wrapper
├── guardrails/
│   ├── g1_preretrieval.py
│   ├── g2_postretrieval.py
│   └── g3_postgeneration.py
├── eval/
│   ├── run_eval.py         # run all queries through G1/G2/G3
│   └── metrics.py          # cost, FPR, FNR + Pareto curve
└── notebooks/              # thin Colab runners (optional)
```

## Setup

```bash
git clone https://github.com/VishGuy2001/ScopeGuard.git
cd ScopeGuard
pip install -r requirements.txt
cp .env.example .env        # then put your real OpenAI key in .env
```

The OpenAI key is loaded from `.env`, which is gitignored. **Never commit the
real key.**

## Running the pipeline

Run these in order — each step's output feeds the next:

```bash
python data/build_kb.py        # 1. build the knowledge base
python rag/index.py            # 2. embed it into a FAISS index
python classifier/train.py     # 3. fine-tune the scope classifier
python eval/run_eval.py         # 4. run all queries through G1/G2/G3
python eval/metrics.py          # 5. score + plot the cost-risk Pareto curve
```

Steps 2, 4, and 5 need the OpenAI key. Steps 1 and 3 do not.

## Notes for the team

- **No GPU needed.** Classifier defaults in `config.py` (small batch, few
  epochs) train in minutes on CPU. Colab's free GPU speeds it up if wanted.
- **`build_kb.py` has a fallback.** If the SEC scrape is blocked, it uses a
  small bundled KB so the pipeline still runs. Get the live scrape working
  before reporting final results.
- **`queries.jsonl` is the starter set** (~48 labeled queries). Expand it to
  300 (150 in-scope / 150 out-of-scope), keeping the schema:
  `{"query": ..., "label": "in_scope"|"out_of_scope", "category": ...}`.
- **Retrieval token cost is estimated** (~4 chars/token) since the embeddings
  API does not report per-call usage. Swap in `tiktoken` for exact counts.

## Workflow

Branch per task, never commit to `main` directly:

```bash
git checkout main && git pull
git checkout -b yourname/task
# ... work, then ...
git add . && git commit -m "message" && git push -u origin yourname/task
```

Open a pull request, have a teammate skim it, merge.

## License

MIT — see [LICENSE](LICENSE).

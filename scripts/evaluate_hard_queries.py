
import json
import re
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from finrag.embedding.model import EmbeddingModel


COLLECTION = "power_safety_children"
DATA_PATH = Path(
    "experiments/retrieval/benchmarks/hard_queries_v1.jsonl"
)
TOP_K = 5

# 1. Load questions
questions = [
    json.loads(line)
    for line in DATA_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

# 2. Load corpus from Qdrant
client = QdrantClient(url="http://127.0.0.1:6333")

records = []
offset = None

while True:
    batch, offset = client.scroll(
        collection_name=COLLECTION,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    records.extend(batch)
    if offset is None:
        break

assert len(records) == 83

def get_id(payload):
    return (
        f'{payload["parent_id"]}:'
        f'{payload["child_index"]}'
    )

corpus_ids = [get_id(r.payload) for r in records]
corpus_texts = [r.payload["text"] for r in records]

# 3. Build BM25
for term in [
    "重大隐患", "重大事故隐患",
    "低电压穿越", "进相试验",
    "乙炔含量", "电力系统稳定器",
    "高温高压汽水系统",
]:
    jieba.add_word(term)

def tokenize(text):
    return [
        word
        for word in jieba.lcut(text.lower())
        if word.strip()
        and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", word)
    ]

bm25 = BM25Okapi(
    [tokenize(text) for text in corpus_texts],
    k1=1.5,
    b=0.75,
)

def search_bm25(query, k):
    scores = bm25.get_scores(tokenize(query))
    indices = np.argsort(-scores, kind="stable")[:k]
    return [corpus_ids[int(i)] for i in indices]

# 4. Dense Retrieval
embedding = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)

query_vectors = embedding.encode_documents(
    [q["query"] for q in questions]
)

def search_dense(vector, k):
    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=k,
        with_payload=True,
    ).points

    return [get_id(r.payload) for r in results]

# 5. Metrics
def metrics(ranked_ids, gold_ids, k):
    retrieved = ranked_ids[:k]
    gold = set(gold_ids)

    hits = gold.intersection(retrieved)

    recall = len(hits) / len(gold)
    all_hit = int(gold.issubset(set(retrieved)))

    ranks = [
        i
        for i, item in enumerate(retrieved, 1)
        if item in gold
    ]

    first_rank = min(ranks) if ranks else None
    rr = 1 / first_rank if first_rank else 0.0

    return {
        "hit": int(bool(hits)),
        "recall": recall,
        "all_hit": all_hit,
        "rr": rr,
    }

# 6. Evaluation
results = []

for q, vector in zip(questions, query_vectors):
    gold = q["gold_children"]

    bm25_ids = search_bm25(q["query"], TOP_K)
    dense_ids = search_dense(vector, TOP_K)

    row = {
        "id": q["id"],
        "category": q["category"],
        "gold": gold,
        "bm25": bm25_ids,
        "dense": dense_ids,
        "metrics": {
            "bm25": metrics(bm25_ids, gold, TOP_K),
            "dense": metrics(dense_ids, gold, TOP_K),
        },
    }
    results.append(row)

    print(f'\n===== {q["id"]} [{q["category"]}] =====')
    print("Query:", q["query"])
    print("Gold:", gold)
    print("BM25:", bm25_ids)
    print("Dense:", dense_ids)

    print(
        "BM25 Evidence Recall@5:",
        row["metrics"]["bm25"]["recall"]
    )
    print(
        "Dense Evidence Recall@5:",
        row["metrics"]["dense"]["recall"]
    )

# 7. Category summary
print("\n===== CATEGORY SUMMARY =====")

categories = sorted(set(q["category"] for q in questions))

for category in categories:
    subset = [
        row for row in results
        if row["category"] == category
    ]

    print(f"\n{category} (n={len(subset)})")

    for method in ["bm25", "dense"]:
        scores = [
            row["metrics"][method]
            for row in subset
        ]

        avg_recall = np.mean(
            [s["recall"] for s in scores]
        )
        avg_all_hit = np.mean(
            [s["all_hit"] for s in scores]
        )
        avg_mrr = np.mean(
            [s["rr"] for s in scores]
        )

        print(
            f"{method}: "
            f"Recall@5={avg_recall:.3f}, "
            f"AllHit@5={avg_all_hit:.3f}, "
            f"MRR@5={avg_mrr:.3f}"
        )

# 8. Save results
output = Path(
    "experiments/retrieval/hard_query_results.json"
)

output.parent.mkdir(parents=True, exist_ok=True)

output.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nSaved:", output)
client.close()

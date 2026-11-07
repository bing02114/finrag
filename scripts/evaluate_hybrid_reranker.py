
import json
import re
import time
from pathlib import Path
from statistics import mean

import jieba
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from finrag.embedding.model import EmbeddingModel


# ============================================
# 1. Experiment Configuration
# ============================================

COLLECTION = "power_safety_children"

DATA_PATH = Path(
    "experiments/retrieval/benchmarks/"
    "hard_queries_v1.jsonl"
)

OUTPUT_PATH = Path(
    "experiments/retrieval/"
    "hybrid_reranker_results.json"
)

RECALL_K = 20
RERANK_K = 20
FINAL_K = 5
RRF_K = 60

RERANKER_NAME = "BAAI/bge-reranker-v2-m3"


# ============================================
# 2. Load Questions and Corpus
# ============================================

questions = [
    json.loads(line)
    for line in DATA_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

client = QdrantClient(
    url="http://127.0.0.1:6333"
)

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


def get_id(payload):
    return (
        f'{payload["parent_id"]}:'
        f'{payload["child_index"]}'
    )


corpus = {
    get_id(record.payload): record.payload["text"]
    for record in records
}

corpus_ids = list(corpus)
corpus_texts = list(corpus.values())

print("Questions:", len(questions))
print("Documents:", len(corpus))

assert len(corpus) == 83


# ============================================
# 3. Build BM25
# ============================================

for term in [
    "重大隐患",
    "重大事故隐患",
    "低电压穿越",
    "进相试验",
    "乙炔含量",
    "电力系统稳定器",
    "高温高压汽水系统",
]:
    jieba.add_word(term)


def tokenize(text):
    return [
        word
        for word in jieba.lcut(text.lower())
        if word.strip()
        and re.search(
            r"[\u4e00-\u9fffA-Za-z0-9]",
            word,
        )
    ]


bm25 = BM25Okapi(
    [tokenize(text) for text in corpus_texts],
    k1=1.5,
    b=0.75,
)


def search_bm25(query, k=20):
    scores = bm25.get_scores(tokenize(query))

    indices = np.argsort(
        -scores,
        kind="stable",
    )[:k]

    return [corpus_ids[int(i)] for i in indices]


# ============================================
# 4. Dense Retriever and Reranker
# ============================================

embedding = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)

reranker = CrossEncoder(
    RERANKER_NAME,
    device="cuda",
    max_length=512,
    activation_fn=torch.nn.Identity(),
)


def search_dense(query_vector, k=20):
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=k,
        with_payload=True,
    ).points

    return [get_id(hit.payload) for hit in hits]


# ============================================
# 5. Reciprocal Rank Fusion
# ============================================

def rrf_fusion(bm25_ids, dense_ids, k=60):
    scores = {}

    for ranked_list in (bm25_ids, dense_ids):
        for rank, doc_id in enumerate(ranked_list, 1):
            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], doc_id),
    )


# ============================================
# 6. Cross-Encoder Reranking
# ============================================

def rerank(query, candidate_ids):
    if not candidate_ids:
        return []

    pairs = [
        (query, corpus[doc_id])
        for doc_id in candidate_ids
    ]

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    return [
        doc_id
        for doc_id, score in sorted(
            zip(candidate_ids, scores),
            key=lambda item: -float(item[1]),
        )
    ]


# ============================================
# 7. Evaluation Metrics
# ============================================

def evaluate(ranked_ids, gold_ids, k=5):
    retrieved = ranked_ids[:k]
    gold = set(gold_ids)

    matched = gold.intersection(retrieved)

    ranks = [
        i
        for i, doc_id in enumerate(retrieved, 1)
        if doc_id in gold
    ]

    return {
        "hit": int(bool(matched)),
        "recall": len(matched) / len(gold),
        "all_hit": int(
            gold.issubset(set(retrieved))
        ),
        "rr": 1 / min(ranks) if ranks else 0.0,
    }


# ============================================
# 8. Encode Queries Once
# ============================================

query_vectors = embedding.encode_documents(
    [q["query"] for q in questions]
)

# Warm up reranker (not counted in per-query latency)
reranker.predict(
    [
        (questions[0]["query"], corpus_texts[0])
    ],
    show_progress_bar=False,
)

torch.cuda.synchronize()


# ============================================
# 9. Run Full Pipeline
# ============================================

results = []

for question, query_vector in zip(
    questions, query_vectors
):

    query = question["query"]
    gold = question["gold_children"]

    bm25_ids = search_bm25(query, RECALL_K)
    dense_ids = search_dense(query_vector, RECALL_K)

    union_ids = set(bm25_ids) | set(dense_ids)

    hybrid_ids = rrf_fusion(
        bm25_ids,
        dense_ids,
        RRF_K,
    )

    candidates = hybrid_ids[:RERANK_K]

    # Measure reranker execution only
    torch.cuda.synchronize()
    start = time.perf_counter()

    reranked_ids = rerank(query, candidates)

    torch.cuda.synchronize()
    rerank_latency = time.perf_counter() - start

    row = {
        "id": question["id"],
        "category": question["category"],
        "query": query,
        "gold": gold,
        "bm25": bm25_ids[:FINAL_K],
        "dense": dense_ids[:FINAL_K],
        "hybrid": hybrid_ids[:FINAL_K],
        "reranked": reranked_ids[:FINAL_K],
        "union_recall": len(
            set(gold) & union_ids
        ) / len(set(gold)),
        "candidate_recall": evaluate(
            candidates, gold, RERANK_K
        )["recall"],
        "rerank_latency": rerank_latency,
    }

    row["metrics"] = {
        "bm25": evaluate(bm25_ids, gold, FINAL_K),
        "dense": evaluate(dense_ids, gold, FINAL_K),
        "hybrid": evaluate(hybrid_ids, gold, FINAL_K),
        "reranked": evaluate(
            reranked_ids, gold, FINAL_K
        ),
    }

    results.append(row)

    print(f'\n===== {question["id"]} =====')
    print("Gold:", gold)
    print("Union Recall:", row["union_recall"])
    print("Candidate Recall:", row["candidate_recall"])
    print("RRF Top-5:", row["hybrid"])
    print("Reranked Top-5:", row["reranked"])
    print(
        "Reranker Latency:",
        round(rerank_latency, 4), "s"
    )


# ============================================
# 10. Final Summary
# ============================================

print("\n===== FINAL SUMMARY =====")

methods = ["bm25", "dense", "hybrid", "reranked"]

for method in methods:
    metrics = [
        row["metrics"][method]
        for row in results
    ]

    print(
        f"{method:9s} "
        f"Hit@1={mean(m['hit'] if m['rr'] == 1.0 else 0 for m in metrics):.3f} "
        f"Recall@5={mean(m['recall'] for m in metrics):.3f} "
        f"AllHit@5={mean(m['all_hit'] for m in metrics):.3f} "
        f"MRR@5={mean(m['rr'] for m in metrics):.3f}"
    )

print("\nUnion Recall@40:",
      round(mean(r["union_recall"] for r in results), 3))

print("RRF Candidate Recall@20:",
      round(mean(r["candidate_recall"] for r in results), 3))

print("Mean Reranker Latency:",
      round(mean(r["rerank_latency"] for r in results), 4),
      "s")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.write_text(
    json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("\nSaved:", OUTPUT_PATH)

client.close()

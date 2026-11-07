
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


# ======================================
# Configuration
# ======================================

COLLECTION = "power_safety_children"

BENCHMARK_PATH = Path(
    "experiments/retrieval/benchmarks/"
    "hard_queries_v1.jsonl"
)

REWRITE_PATH = Path(
    "experiments/retrieval/query_rewrite_v1.json"
)

OUTPUT_PATH = Path(
    "experiments/retrieval/query_rewrite_comparison.json"
)

TOP_K_RECALL = 20
TOP_K_CANDIDATE = 20
FINAL_K = 5
RRF_K = 60


# ======================================
# 1. Load questions and rewrites
# ======================================

questions = [
    json.loads(line)
    for line in BENCHMARK_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

rewrites = json.loads(
    REWRITE_PATH.read_text(encoding="utf-8")
)

rewrite_map = {
    item["id"]: item
    for item in rewrites
}

assert len(questions) == len(rewrites)


# ======================================
# 2. Load corpus from Qdrant
# ======================================

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

print("Corpus:", len(corpus))
assert len(corpus) == 83


# ======================================
# 3. Build BM25
# ======================================

for term in [
    "重大隐患",
    "低电压穿越",
    "进相试验",
    "乙炔含量",
    "电力系统稳定器",
    "高温高压汽水系统",
]:
    jieba.add_word(term)


def tokenize(text):
    return [
        token
        for token in jieba.lcut(text.lower())
        if token.strip()
        and re.search(
            r"[\u4e00-\u9fffA-Za-z0-9]",
            token,
        )
    ]


corpus_ids = list(corpus)
corpus_texts = list(corpus.values())

bm25 = BM25Okapi(
    [tokenize(text) for text in corpus_texts],
    k1=1.5,
    b=0.75,
)


def search_bm25(query):
    scores = bm25.get_scores(tokenize(query))

    indices = np.argsort(
        -scores,
        kind="stable",
    )[:TOP_K_RECALL]

    return [corpus_ids[int(i)] for i in indices]


# ======================================
# 4. Initialize models
# ======================================

embedding = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cuda",
    max_length=512,
    activation_fn=torch.nn.Identity(),
)


# ======================================
# 5. Retrieval and fusion
# ======================================

def search_dense(query):
    vector = embedding.encode_query(query)

    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=TOP_K_RECALL,
        with_payload=True,
    ).points

    return [get_id(hit.payload) for hit in hits]


def rrf(bm25_ids, dense_ids):
    scores = {}

    for ranked in [bm25_ids, dense_ids]:
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] = (
                scores.get(doc_id, 0)
                + 1 / (RRF_K + rank)
            )

    return sorted(
        scores,
        key=lambda x: (-scores[x], x),
    )


def rerank(query, candidates):
    pairs = [
        (query, corpus[doc_id])
        for doc_id in candidates
    ]

    torch.cuda.synchronize()
    start = time.perf_counter()

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    torch.cuda.synchronize()

    latency = time.perf_counter() - start

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: -float(x[1]),
    )

    return [doc_id for doc_id, _ in ranked], latency


# ======================================
# 6. Metrics
# ======================================

def evaluate(ranked, gold, k=5):
    top = ranked[:k]
    gold = set(gold)

    matched = gold.intersection(top)

    ranks = [
        i
        for i, doc_id in enumerate(top, 1)
        if doc_id in gold
    ]

    return {
        "hit1": int(bool(top) and top[0] in gold),
        "recall": len(matched) / len(gold),
        "all_hit": int(gold.issubset(set(top))),
        "mrr": 1 / min(ranks) if ranks else 0.0,
    }


# ======================================
# 7. Run paired experiments
# ======================================

results = []

# Warm-up
reranker.predict(
    [("test query", corpus_texts[0])],
    show_progress_bar=False,
)
torch.cuda.synchronize()


for question in questions:

    qid = question["id"]
    gold = question["gold_children"]

    original = question["query"]

    rewritten = rewrite_map[qid][
        "rewritten_query"
    ]

    print(f"\n===== {qid} =====")

    row = {
        "id": qid,
        "category": question["category"],
        "gold": gold,
        "original": {},
        "rewritten": {},
    }

    for mode, query in [
        ("original", original),
        ("rewritten", rewritten),
    ]:

        bm25_ids = search_bm25(query)
        dense_ids = search_dense(query)

        union = set(bm25_ids) | set(dense_ids)

        candidates = rrf(
            bm25_ids,
            dense_ids,
        )[:TOP_K_CANDIDATE]

        reranked, latency = rerank(
            query,
            candidates,
        )

        row[mode] = {
            "query": query,
            "union_recall": (
                len(set(gold) & union) / len(set(gold))
            ),
            "candidate_recall": evaluate(
                candidates, gold, TOP_K_CANDIDATE
            )["recall"],
            "final": reranked[:FINAL_K],
            "metrics": evaluate(
                reranked, gold, FINAL_K
            ),
            "rerank_latency": latency,
        }

        print(f"\n{mode.upper()}:")
        print("Query:", query)
        print("Top-5:", reranked[:FINAL_K])
        print("Metrics:", row[mode]["metrics"])

    results.append(row)


# ======================================
# 8. Summary
# ======================================

print("\n===== FINAL COMPARISON =====")

for mode in ["original", "rewritten"]:

    rows = [r[mode] for r in results]

    print(f"\n{mode.upper()}")

    for metric in [
        "hit1",
        "recall",
        "all_hit",
        "mrr",
    ]:
        value = mean(
            row["metrics"][metric]
            for row in rows
        )

        print(f"{metric}: {value:.4f}")

    print(
        "Union Recall@40:",
        round(mean(
            row["union_recall"]
            for row in rows
        ), 4),
    )

    print(
        "Candidate Recall@20:",
        round(mean(
            row["candidate_recall"]
            for row in rows
        ), 4),
    )

    print(
        "Reranker Latency:",
        round(mean(
            row["rerank_latency"]
            for row in rows
        ), 4),
        "s",
    )


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

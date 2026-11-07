
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
# 1. Configuration
# ======================================

COLLECTION = "power_safety_children"

BENCHMARK_PATH = Path(
    "experiments/retrieval/benchmarks/"
    "hard_queries_v1.jsonl"
)

DECOMPOSITION_PATH = Path(
    "experiments/retrieval/"
    "query_decomposition_v1.json"
)

OUTPUT_PATH = Path(
    "experiments/retrieval/"
    "decomposition_results.json"
)

RECALL_K = 20
CANDIDATE_K = 20
FINAL_K = 5
RRF_K = 60


# ======================================
# 2. Load Questions and Decomposition
# ======================================

questions = [
    json.loads(line)
    for line in BENCHMARK_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

decompositions = json.loads(
    DECOMPOSITION_PATH.read_text(
        encoding="utf-8"
    )
)

decomposition_map = {
    item["id"]: item
    for item in decompositions
}

questions = [
    q for q in questions
    if q["category"] == "multi_evidence"
]


# ======================================
# 3. Load Corpus
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
    get_id(r.payload): r.payload["text"]
    for r in records
}

print("Corpus:", len(corpus))

assert len(corpus) == 83


# ======================================
# 4. Build BM25
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
    )[:RECALL_K]

    return [corpus_ids[int(i)] for i in indices]


# ======================================
# 5. Initialize Dense and Reranker
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


def search_dense(query):
    vector = embedding.encode_query(query)

    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=RECALL_K,
        with_payload=True,
    ).points

    return [get_id(hit.payload) for hit in hits]


# ======================================
# 6. Multi-list RRF
# ======================================

def multi_rrf(ranked_lists, k=60):

    scores = {}

    for ranked_list in ranked_lists:

        for rank, doc_id in enumerate(
            ranked_list, 1
        ):

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores,
        key=lambda doc_id: (
            -scores[doc_id],
            doc_id,
        ),
    )


# ======================================
# 7. Multi-query Reranking
# ======================================

def multi_query_rerank(queries, candidates):

    if not candidates:
        return []

    pairs = [
        (query, corpus[doc_id])
        for doc_id in candidates
        for query in queries
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

    # Reshape:
    # [candidate_count, query_count]
    scores = np.asarray(scores).reshape(
        len(candidates),
        len(queries),
    )

    # Best matching sub-query score
    final_scores = scores.max(axis=1)

    ranked = sorted(
        zip(candidates, final_scores),
        key=lambda item: -float(item[1]),
    )

    return (
        [doc_id for doc_id, _ in ranked],
        latency,
    )


# ======================================
# 8. Single-query Baseline
# ======================================

def single_query_pipeline(query, gold):

    bm25_ids = search_bm25(query)
    dense_ids = search_dense(query)

    union = set(bm25_ids) | set(dense_ids)

    hybrid_ids = multi_rrf(
        [bm25_ids, dense_ids],
        RRF_K,
    )

    candidates = hybrid_ids[:CANDIDATE_K]

    reranked, latency = multi_query_rerank(
        [query],
        candidates,
    )

    return {
        "union_recall": (
            len(set(gold) & union) / len(set(gold))
        ),
        "candidate_recall": (
            len(set(gold) & set(candidates))
            / len(set(gold))
        ),
        "final": reranked[:FINAL_K],
        "latency": latency,
        "candidate_count": len(candidates),
    }


# ======================================
# 9. Decomposition Pipeline
# ======================================

def decomposition_pipeline(queries, gold):

    ranked_lists = []
    union = set()

    for query in queries:

        bm25_ids = search_bm25(query)
        dense_ids = search_dense(query)

        ranked_lists.extend([
            bm25_ids,
            dense_ids,
        ])

        union.update(bm25_ids)
        union.update(dense_ids)

    # Fuse all four ranking lists
    hybrid_ids = multi_rrf(
        ranked_lists,
        RRF_K,
    )

    candidates = hybrid_ids[:CANDIDATE_K]

    # Score every candidate against every sub-query
    reranked, latency = multi_query_rerank(
        queries,
        candidates,
    )

    return {
        "union_recall": (
            len(set(gold) & union) / len(set(gold))
        ),
        "candidate_recall": (
            len(set(gold) & set(candidates))
            / len(set(gold))
        ),
        "final": reranked[:FINAL_K],
        "latency": latency,
        "candidate_count": len(candidates),
    }


# ======================================
# 10. Metrics
# ======================================

def evaluate(ranked, gold):

    top = ranked[:FINAL_K]
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
# 11. Run Experiments
# ======================================

results = []

# Warm-up
reranker.predict(
    [("warm up", corpus_texts[0])],
    show_progress_bar=False,
)
torch.cuda.synchronize()


for question in questions:

    qid = question["id"]
    original = question["query"]
    gold = question["gold_children"]

    queries = decomposition_map[qid]["queries"]

    print(f"\n===== {qid} =====")
    print("Original:", original)

    print("Sub-queries:")
    for query in queries:
        print(" -", query)

    # Baseline
    baseline = single_query_pipeline(
        original,
        gold,
    )

    # Decomposition
    decomposed = decomposition_pipeline(
        queries,
        gold,
    )

    row = {
        "id": qid,
        "gold": gold,
        "queries": queries,
        "baseline": baseline,
        "decomposed": decomposed,
        "baseline_metrics": evaluate(
            baseline["final"], gold
        ),
        "decomposed_metrics": evaluate(
            decomposed["final"], gold
        ),
    }

    results.append(row)

    print("\nBaseline:")
    print("Top-5:", baseline["final"])
    print("Metrics:", row["baseline_metrics"])

    print("\nDecomposed:")
    print("Top-5:", decomposed["final"])
    print("Metrics:", row["decomposed_metrics"])

    print(
        "Candidate Recall:",
        baseline["candidate_recall"],
        "->",
        decomposed["candidate_recall"],
    )


# ======================================
# 12. Final Summary
# ======================================

print("\n===== FINAL SUMMARY =====")

for name, key in [
    ("Baseline", "baseline_metrics"),
    ("Decomposition", "decomposed_metrics"),
]:

    metrics = [r[key] for r in results]

    print(f"\n{name}")

    for metric in [
        "hit1",
        "recall",
        "all_hit",
        "mrr",
    ]:
        print(
            metric,
            round(
                mean(m[metric] for m in metrics),
                4,
            ),
        )


print("\n===== CANDIDATE COVERAGE =====")

for field in [
    "union_recall",
    "candidate_recall",
    "latency",
]:
    baseline_values = [
        r["baseline"][field]
        for r in results
    ]

    decomposed_values = [
        r["decomposed"][field]
        for r in results
    ]

    print(
        field,
        "Baseline:",
        round(mean(baseline_values), 4),
        "Decomposed:",
        round(mean(decomposed_values), 4),
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
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

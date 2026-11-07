
import re

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from finrag.embedding.model import EmbeddingModel


COLLECTION = "power_safety_children"
TOP_K = 5

client = QdrantClient(
    url="http://127.0.0.1:6333"
)


# =====================================
# 1. Load all child records from Qdrant
# =====================================

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

print("Loaded child records:", len(records))

assert len(records) == 83, (
    f"Expected 83 records, got {len(records)}"
)


# =====================================
# 2. Chinese tokenization
# =====================================

def tokenize(text: str) -> list[str]:
    text = text.lower()

    tokens = jieba.lcut(text)

    return [
        token
        for token in tokens
        if token.strip()
        and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", token)
    ]


# Add some domain terminology
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

# Tokenize documents
corpus = [
    record.payload["text"]
    for record in records
]

tokenized_corpus = [
    tokenize(text)
    for text in corpus
]

print("\n===== TOKENIZATION =====")

example = "变压器乙炔含量超过多少属于重大隐患？"

print("Text:", example)
print("Tokens:", tokenize(example))


# =====================================
# 3. Build BM25 index
# =====================================

bm25 = BM25Okapi(
    tokenized_corpus,
    k1=1.5,
    b=0.75,
)


def bm25_search(query: str, k: int = 5):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(
        -scores,
        kind="stable",
    )

    results = []

    for index in ranked_indices[:k]:
        record = records[int(index)]

        results.append({
            "parent_id": record.payload["parent_id"],
            "child_index": record.payload["child_index"],
            "text": record.payload["text"],
            "score": float(scores[index]),
        })

    return results


# =====================================
# 4. Dense Retrieval
# =====================================

embedding = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)


def dense_search(query_vector, k: int = 5):
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=k,
        with_payload=True,
    ).points

    return [
        {
            "parent_id": hit.payload["parent_id"],
            "child_index": hit.payload["child_index"],
            "text": hit.payload["text"],
            "score": float(hit.score),
        }
        for hit in hits
    ]


# =====================================
# 5. Evaluation Questions
# =====================================

questions = [
    {
        "id": "Q01",
        "query": "变压器乙炔含量达到多少构成重大隐患？",
        "gold_parent": "D01_第五条",
        "gold_child": 5,
    },
    {
        "id": "Q02",
        "query": "变压器基础沉降差超过什么标准属于重大隐患？",
        "gold_parent": "D01_第五条",
        "gold_child": 6,
    },
    {
        "id": "Q03",
        "query": "风电场不满足低电压穿越能力有什么安全风险判定？",
        "gold_parent": "D01_第五条",
        "gold_child": 4,
    },
    {
        "id": "Q04",
        "query": "哪些发电机组未开展进相试验属于重大隐患？",
        "gold_parent": "D01_第五条",
        "gold_child": 3,
    },
    {
        "id": "Q05",
        "query": "高温高压汽水系统泄漏时带压堵漏属于什么隐患？",
        "gold_parent": "D01_第六条",
        "gold_child": 2,
    },
    {
        "id": "Q06",
        "query": "汽轮机调节部套卡涩是否构成重大隐患？",
        "gold_parent": "D01_第六条",
        "gold_child": 0,
    },
]


# =====================================
# 6. Metrics
# =====================================

def get_rank(results, gold_parent, gold_child):
    for rank, item in enumerate(results, start=1):
        if (
            item["parent_id"] == gold_parent
            and item["child_index"] == gold_child
        ):
            return rank

    return None


def evaluate(ranks, k):
    hit = sum(
        rank is not None and rank <= k
        for rank in ranks
    ) / len(ranks)

    mrr = sum(
        1 / rank
        for rank in ranks
        if rank is not None and rank <= k
    ) / len(ranks)

    return hit, mrr


# =====================================
# 7. Run Comparison
# =====================================

query_vectors = embedding.encode_documents(
    [q["query"] for q in questions]
)

bm25_ranks = []
dense_ranks = []

for question, query_vector in zip(
    questions, query_vectors
):
    bm25_results = bm25_search(
        question["query"], TOP_K
    )

    dense_results = dense_search(
        query_vector, TOP_K
    )

    bm25_rank = get_rank(
        bm25_results,
        question["gold_parent"],
        question["gold_child"],
    )

    dense_rank = get_rank(
        dense_results,
        question["gold_parent"],
        question["gold_child"],
    )

    bm25_ranks.append(bm25_rank)
    dense_ranks.append(dense_rank)

    print(f'\n===== {question["id"]} =====')
    print("Query:", question["query"])
    print("Gold:", question["gold_parent"],
          question["gold_child"])

    print("BM25 Rank:", bm25_rank)
    print("Dense Rank:", dense_rank)

    for name, results in [
        ("BM25", bm25_results),
        ("Dense", dense_results),
    ]:
        print(f"\n{name} Top-{TOP_K}:")

        for rank, item in enumerate(results, 1):
            print(
                rank,
                item["parent_id"],
                item["child_index"],
                f'score={item["score"]:.4f}',
            )


# =====================================
# 8. Final Metrics
# =====================================

print("\n===== FINAL METRICS =====")

for name, ranks in [
    ("BM25", bm25_ranks),
    ("Dense", dense_ranks),
]:
    print(f"\n{name}")

    for k in [1, 3, 5]:
        hit, mrr = evaluate(ranks, k)

        print(
            f"Hit@{k}: {hit:.4f}, "
            f"MRR@{k}: {mrr:.4f}"
        )

client.close()

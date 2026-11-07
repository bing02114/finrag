
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from finrag.ingestion.loader import DocumentLoader
from finrag.chunking.structure_chunker import StructureChunker
from finrag.embedding.model import EmbeddingModel


PDF_PATH = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw/pdf/"
    "D01_power_hazard_regulation.pdf"
)

ARTICLE_COLLECTION = "power_safety_articles"
CHILD_COLLECTION = "power_safety_children"

TOP_K = 3


# ======================================
# 1. Load the same document
# ======================================

pages = DocumentLoader.load_pdf(str(PDF_PATH), "D01")
articles = StructureChunker.split_articles(pages)

print("Articles:", len(articles))


# ======================================
# 2. Load embedding model once
# ======================================

embedding = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)

client = QdrantClient(url="http://127.0.0.1:6333")


# ======================================
# 3. Build Article-level index
# ======================================

if not client.collection_exists(ARTICLE_COLLECTION):
    client.create_collection(
        collection_name=ARTICLE_COLLECTION,
        vectors_config=models.VectorParams(
            size=embedding.get_dimension(),
            distance=models.Distance.COSINE,
        ),
    )

vectors = embedding.encode_documents(
    [article["text"] for article in articles]
)

points = []

for article, vector in zip(articles, vectors):
    parent_id = (
        f'{article["document_id"]}_{article["article"]}'
    )

    points.append(
        models.PointStruct(
            id=str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                parent_id,
            )),
            vector=vector,
            payload={
                **article,
                "parent_id": parent_id,
            },
        )
    )

client.upload_points(
    collection_name=ARTICLE_COLLECTION,
    points=points,
    batch_size=32,
    wait=True,
)

article_count = client.count(
    ARTICLE_COLLECTION, exact=True
).count

child_count = client.count(
    CHILD_COLLECTION, exact=True
).count

print("Indexed Articles:", article_count)
print("Indexed Children:", child_count)

assert article_count == 38
assert child_count == 83


# ======================================
# 4. Small evaluation set
# ======================================

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


# ======================================
# 5. Retrieval function
# ======================================

def search(collection, query_vector, limit=3):
    return client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
    ).points


def find_parent_rank(results, gold_parent):
    for rank, result in enumerate(results, start=1):
        if result.payload["parent_id"] == gold_parent:
            return rank
    return None


def find_child_rank(results, gold_parent, gold_child):
    for rank, result in enumerate(results, start=1):
        payload = result.payload

        if (
            payload["parent_id"] == gold_parent
            and payload["child_index"] == gold_child
        ):
            return rank

    return None


def reciprocal_rank(rank):
    return 1.0 / rank if rank is not None else 0.0


# ======================================
# 6. Compare retrieval results
# ======================================

article_ranks = []
child_parent_ranks = []
exact_child_ranks = []

for question in questions:

    query_vector = embedding.encode_query(
        question["query"]
    )

    article_results = search(
        ARTICLE_COLLECTION, query_vector, TOP_K
    )

    child_results = search(
        CHILD_COLLECTION, query_vector, TOP_K
    )

    article_rank = find_parent_rank(
        article_results,
        question["gold_parent"],
    )

    child_parent_rank = find_parent_rank(
        child_results,
        question["gold_parent"],
    )

    exact_child_rank = find_child_rank(
        child_results,
        question["gold_parent"],
        question["gold_child"],
    )

    article_ranks.append(article_rank)
    child_parent_ranks.append(child_parent_rank)
    exact_child_ranks.append(exact_child_rank)

    print(f'\n===== {question["id"]} =====')
    print("Query:", question["query"])
    print("Article rank:", article_rank)
    print("Child -> Parent rank:", child_parent_rank)
    print("Exact Child rank:", exact_child_rank)

    print("Article Top-3:", [
        r.payload["article"]
        for r in article_results
    ])

    print("Child Top-3:", [
        (
            r.payload["article"],
            r.payload["child_index"],
        )
        for r in child_results
    ])


# ======================================
# 7. Evaluation metrics
# ======================================

def hit_at_k(ranks, k):
    return sum(
        rank is not None and rank <= k
        for rank in ranks
    ) / len(ranks)


def mrr(ranks):
    return sum(
        reciprocal_rank(rank)
        for rank in ranks
    ) / len(ranks)


print("\n===== FINAL METRICS =====")

for name, ranks in [
    ("Article", article_ranks),
    ("Child -> Parent", child_parent_ranks),
    ("Exact Child", exact_child_ranks),
]:
    print(f"\n{name}")
    print("Hit@1:", round(hit_at_k(ranks, 1), 4))
    print("Hit@3:", round(hit_at_k(ranks, 3), 4))
    print("MRR@3:", round(mrr(ranks), 4))

client.close()

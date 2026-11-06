
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from finrag.embedding.model import EmbeddingModel


# 1. Connect to Server
client = QdrantClient(
    url="http://127.0.0.1:6333"
)

collection = "financial_reports_server"


# 2. Load Embedding Model
embedding_model = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)


# 3. Prepare Documents
documents = [
    {
        "id": 1,
        "text": "贵州茅台2024年营业收入为1741.44亿元。",
        "company_name": "贵州茅台",
        "stock_code": "600519",
        "report_year": 2024,
        "page_number": 12,
        "section": "经营情况",
    },
    {
        "id": 2,
        "text": "贵州茅台2023年营业收入为1476.94亿元。",
        "company_name": "贵州茅台",
        "stock_code": "600519",
        "report_year": 2023,
        "page_number": 11,
        "section": "经营情况",
    },
    {
        "id": 3,
        "text": "五粮液2024年营业收入为891.75亿元。",
        "company_name": "五粮液",
        "stock_code": "000858",
        "report_year": 2024,
        "page_number": 15,
        "section": "经营情况",
    },
    {
        "id": 4,
        "text": "贵州茅台2024年研发费用持续投入。",
        "company_name": "贵州茅台",
        "stock_code": "600519",
        "report_year": 2024,
        "page_number": 20,
        "section": "研发情况",
    },
]


# 4. Generate Embeddings
vectors = embedding_model.encode_documents(
    [doc["text"] for doc in documents]
)


# 5. Upsert Points
points = []

for doc, vector in zip(documents, vectors):

    payload = {
        key: value
        for key, value in doc.items()
        if key != "id"
    }

    points.append(
        PointStruct(
            id=doc["id"],
            vector=vector,
            payload=payload,
        )
    )

client.upsert(
    collection_name=collection,
    points=points,
    wait=True,
)

print("Inserted:", len(points))


# 6. Dense Retrieval
query = "贵州茅台2024年的营业收入是多少？"

query_vector = embedding_model.encode_query(query)

results = client.query_points(
    collection_name=collection,
    query=query_vector,
    limit=3,
).points

print("\n===== DENSE SEARCH =====")

for result in results:
    print(
        result.id,
        round(result.score, 4),
        result.payload["text"],
    )


# 7. Filtered Retrieval
results = client.query_points(
    collection_name=collection,
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="stock_code",
                match=MatchValue(value="600519"),
            ),
            FieldCondition(
                key="report_year",
                match=MatchValue(value=2024),
            ),
        ]
    ),
    limit=3,
).points

print("\n===== FILTERED SEARCH =====")

for result in results:
    print(
        result.id,
        round(result.score, 4),
        result.payload["text"],
    )


# 8. Inspect Indexes
info = client.get_collection(collection)

print("\n===== FINAL COLLECTION =====")
print("Points:", info.points_count)
print("Payload schema:", info.payload_schema)

client.close()

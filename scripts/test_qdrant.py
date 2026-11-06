
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from finrag.embedding.model import EmbeddingModel


# ==================================
# 1. Initialize Embedding Model
# ==================================

embedding_model = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)


# ==================================
# 2. Initialize Qdrant
# ==================================

# client = QdrantClient(
#     path="/root/finrag-storage/qdrant"
# )

client = QdrantClient(
    url="http://127.0.0.1:6333"
)

collection_name = "financial_reports"


# ==================================
# 3. Create Collection
# ==================================

if not client.collection_exists(collection_name):

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_model.get_dimension(),
            distance=Distance.COSINE,
        ),
    )

print("Collection ready!")


# ==================================
# 4. Prepare Documents
# ==================================

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


# ==================================
# 5. Generate Embeddings
# ==================================

texts = [doc["text"] for doc in documents]

vectors = embedding_model.encode_documents(texts)


# ==================================
# 6. Insert Points
# ==================================

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
    collection_name=collection_name,
    points=points,
)

print(f"Inserted {len(points)} points!")


# ==================================
# 7. Dense Retrieval
# ==================================

query = "贵州茅台2024年的营业收入是多少？"

query_vector = embedding_model.encode_query(query)

results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3,
).points

print("\n===== SEARCH RESULTS =====")

for result in results:

    print("\nScore:", round(result.score, 4))
    print("Text:", result.payload["text"])
    print("Company:", result.payload["company_name"])
    print("Year:", result.payload["report_year"])


# ==================================
# 8. Metadata Filtering
# ==================================

print("\n===== FILTERED SEARCH =====")

results = client.query_points(
    collection_name=collection_name,
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

for result in results:

    print("\nScore:", round(result.score, 4))
    print("Text:", result.payload["text"])
    print("Year:", result.payload["report_year"])


# ==================================
# 9. Collection Information
# ==================================

info = client.get_collection(collection_name)

print("\n===== COLLECTION INFO =====")
print("Points count:", info.points_count)
print("Vector size:", info.config.params.vectors.size)

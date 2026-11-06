
import uuid
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
)

from finrag.embedding.model import EmbeddingModel


COLLECTION = "financial_reports_batch"

# 1. Connect to Qdrant Server
client = QdrantClient(
    url="http://127.0.0.1:6333"
)

# 2. Load embedding model
embedding_model = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)

# 3. Create collection
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=embedding_model.get_dimension(),
        distance=Distance.COSINE,
    ),
)

# 4. Create payload indexes before inserting data
indexes = {
    "document_id": PayloadSchemaType.KEYWORD,
    "document_version": PayloadSchemaType.KEYWORD,
    "stock_code": PayloadSchemaType.KEYWORD,
    "report_year": PayloadSchemaType.INTEGER,
}

for field, schema in indexes.items():
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name=field,
        field_schema=schema,
        wait=True,
    )


# 5. Generate deterministic UUID5
def generate_chunk_id(
    document_id: str,
    document_version: str,
    chunk_index: int,
) -> str:

    identity = (
        f"{document_id}:"
        f"{document_version}:"
        f"{chunk_index}"
    )

    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, identity)
    )


# 6. Prepare documents
document_id = "600519_2024_annual_report"

texts_v1 = [
    "贵州茅台2024年营业收入为1741.44亿元。",
    "贵州茅台2024年净利润为862.28亿元。",
    "贵州茅台2024年持续加大研发投入。",
    "贵州茅台2024年经营活动现金流保持稳定。",
]

texts_v2 = [
    "贵州茅台2024年营业收入为1741.44亿元。",
    "贵州茅台2024年净利润为862.28亿元。",
    "贵州茅台2024年持续加大研发投入。",
    "贵州茅台2024年经营活动现金流保持稳定。",
    "贵州茅台2024年加强数字化建设。",
]


# 7. Build points in batches
def build_points(
    texts,
    version,
    batch_size=2,
):

    points = []

    for start in range(0, len(texts), batch_size):

        batch_texts = texts[start:start + batch_size]

        # Batch Embedding
        vectors = embedding_model.encode_documents(
            batch_texts
        )

        for offset, (text, vector) in enumerate(
            zip(batch_texts, vectors)
        ):

            index = start + offset

            point_id = generate_chunk_id(
                document_id,
                version,
                index,
            )

            payload = {
                "document_id": document_id,
                "document_version": version,
                "stock_code": "600519",
                "report_year": 2024,
                "chunk_index": index,
                "text": text,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

    return points


# 8. Batch Upsert
def batch_upsert(points, batch_size=2):

    for start in range(0, len(points), batch_size):

        batch = points[start:start + batch_size]

        client.upsert(
            collection_name=COLLECTION,
            points=batch,
            wait=True,
        )

        print(
            f"Uploaded {start + len(batch)}/{len(points)}"
        )


# ======================================
# Experiment 1: Insert v1
# ======================================

print("\n===== INSERT V1 =====")

points_v1 = build_points(texts_v1, "v1")

batch_upsert(points_v1)

info = client.get_collection(COLLECTION)
print("Points:", info.points_count)


# ======================================
# Experiment 2: Repeat Upsert
# ======================================

print("\n===== REPEAT UPSERT =====")

batch_upsert(points_v1)

info = client.get_collection(COLLECTION)
print("Points after repeat:", info.points_count)


# ======================================
# Experiment 3: Insert v2
# ======================================

print("\n===== INSERT V2 =====")

points_v2 = build_points(texts_v2, "v2")

batch_upsert(points_v2)

info = client.get_collection(COLLECTION)
print("Points after v2:", info.points_count)


# ======================================
# Experiment 4: Delete old version
# ======================================

print("\n===== DELETE V1 =====")

client.delete(
    collection_name=COLLECTION,
    points_selector=Filter(
        must=[
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            ),
            FieldCondition(
                key="document_version",
                match=MatchValue(value="v1"),
            ),
        ]
    ),
    wait=True,
)

info = client.get_collection(COLLECTION)

print("Final points:", info.points_count)


# ======================================
# Inspect remaining points
# ======================================

points, _ = client.scroll(
    collection_name=COLLECTION,
    limit=100,
    with_payload=True,
    with_vectors=False,
)

print("\n===== FINAL DATA =====")

for point in points:
    print(
        point.id,
        point.payload["document_version"],
        point.payload["text"],
    )

client.close()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)

# Connect to Qdrant Server
client = QdrantClient(
    url="http://127.0.0.1:6333"
)

collection_name = "financial_reports_server"

# Create Collection
if not client.collection_exists(collection_name):

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE,
        ),
    )

# Create Payload Indexes
indexes = {
    "stock_code": PayloadSchemaType.KEYWORD,
    "company_name": PayloadSchemaType.KEYWORD,
    "report_year": PayloadSchemaType.INTEGER,
    "page_number": PayloadSchemaType.INTEGER,
    "section": PayloadSchemaType.KEYWORD,
}

for field, schema in indexes.items():

    client.create_payload_index(
        collection_name=collection_name,
        field_name=field,
        field_schema=schema,
        wait=True,
    )

    print(f"Created index: {field}")

# Inspect Collection
info = client.get_collection(collection_name)

print("\n===== COLLECTION INFO =====")
print("Points:", info.points_count)
print("Status:", info.status)

print("\n===== PAYLOAD SCHEMA =====")
print(info.payload_schema)

client.close()

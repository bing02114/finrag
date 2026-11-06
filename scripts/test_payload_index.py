
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

# 1. Connect to existing local Qdrant
client = QdrantClient(
    path="/root/finrag-storage/qdrant"
)

collection_name = "financial_reports"


# 2. Define payload indexes
indexes = {
    "stock_code": PayloadSchemaType.KEYWORD,
    "company_name": PayloadSchemaType.KEYWORD,
    "report_year": PayloadSchemaType.INTEGER,
    "page_number": PayloadSchemaType.INTEGER,
    "section": PayloadSchemaType.KEYWORD,
}

# 3. Create indexes
for field_name, field_type in indexes.items():

    client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=field_type,
        wait=True,
    )

    print(f"Created index: {field_name}")


# 4. Inspect collection schema
info = client.get_collection(collection_name)

print("\n===== PAYLOAD SCHEMA =====")

print(info.payload_schema)


# 5. Test exact matching
print("\n===== EXACT MATCH =====")

results = client.scroll(
    collection_name=collection_name,
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key="stock_code",
                match=MatchValue(value="600519"),
            )
        ]
    ),
    limit=10,
    with_payload=True,
    with_vectors=False,
)[0]

for point in results:
    print(
        point.id,
        point.payload["text"],
        point.payload["stock_code"],
    )


# 6. Test range filtering
print("\n===== RANGE FILTER =====")

results = client.scroll(
    collection_name=collection_name,
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key="report_year",
                range=Range(
                    gte=2023,
                    lte=2024,
                ),
            )
        ]
    ),
    limit=10,
    with_payload=True,
    with_vectors=False,
)[0]

for point in results:
    print(
        point.id,
        point.payload["text"],
        point.payload["report_year"],
    )


# 7. Test combined conditions
print("\n===== COMBINED FILTER =====")

results = client.scroll(
    collection_name=collection_name,
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key="stock_code",
                match=MatchValue(value="600519"),
            ),
            FieldCondition(
                key="report_year",
                range=Range(
                    gte=2024,
                ),
            ),
        ]
    ),
    limit=10,
    with_payload=True,
    with_vectors=False,
)[0]

for point in results:
    print(point.payload["text"])

client.close()

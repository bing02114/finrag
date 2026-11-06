from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://127.0.0.1:6333"
)

original = client.get_collection(
    "financial_reports_batch"
)

restored = client.get_collection(
    "financial_reports_batch_restore"
)

print("Original:", original.points_count)
print("Restored:", restored.points_count)

assert original.points_count == restored.points_count

print("Snapshot recovery successful!")

points, _ = client.scroll(
    collection_name="financial_reports_batch_restore",
    limit=10,
    with_payload=True,
    with_vectors=False,
)

for point in points:
    print(
        point.payload["document_id"],
        point.payload["document_version"],
        point.payload["text"],
    )
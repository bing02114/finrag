
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from finrag.ingestion.loader import DocumentLoader
from finrag.chunking.structure_chunker import StructureChunker
from finrag.chunking.parent_child import split_article_children
from finrag.embedding.model import EmbeddingModel


PDF_PATH = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw/pdf/"
    "D01_power_hazard_regulation.pdf"
)

COLLECTION = "power_safety_children"

# 1. Load document
pages = DocumentLoader.load_pdf(
    str(PDF_PATH), "D01"
)

# 2. Extract parent articles
parents = StructureChunker.split_articles(pages)

# Parent lookup table
parent_store = {
    f'{p["document_id"]}_{p["article"]}': p
    for p in parents
}

# 3. Split into children
children = []

for parent in parents:
    children.extend(
        split_article_children(parent)
    )

print("Parents:", len(parents))
print("Children:", len(children))

# 4. Initialize embedding model
embedding = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=32,
)

# 5. Connect to Qdrant
client = QdrantClient(
    url="http://127.0.0.1:6333"
)

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=embedding.get_dimension(),
            distance=Distance.COSINE,
        ),
    )

# 6. Batch embedding
vectors = embedding.encode_documents(
    [child["text"] for child in children]
)

# 7. Construct points
points = []

for child, vector in zip(children, vectors):

    identity = (
        f'{child["parent_id"]}:'
        f'{child["child_index"]}'
    )

    point_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, identity)
    )

    points.append(
        PointStruct(
            id=point_id,
            vector=vector,
            payload=child,
        )
    )

# 8. Upload to Qdrant
client.upload_points(
    collection_name=COLLECTION,
    points=points,
    batch_size=64,
    wait=True,
)

print("Indexed children:", client.count(
    collection_name=COLLECTION,
    exact=True
).count)

# 9. Query
query = "变压器乙炔含量达到多少属于重大事故隐患？"

query_vector = embedding.encode_query(query)

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=3,
).points

print("\n===== CHILD RETRIEVAL =====")

for result in results:
    payload = result.payload
    print("\nScore:", round(result.score, 4))
    print("Parent ID:", payload["parent_id"])
    print("Child:", payload["text"][:250])

# 10. Parent lookup (deduplicated)
print("\n===== PARENT RETRIEVAL =====")

seen = set()

for result in results:
    parent_id = result.payload["parent_id"]

    if parent_id in seen:
        continue

    seen.add(parent_id)

    parent = parent_store[parent_id]

    print("\nParent:", parent_id)
    print("Chapter:", parent["chapter"])
    print(
        "Pages:",
        parent["start_page"],
        "-",
        parent["end_page"]
    )
    print("Full article:")
    print(parent["text"][:1200])

client.close()

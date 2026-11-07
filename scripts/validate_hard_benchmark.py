import json
from pathlib import Path
from qdrant_client import QdrantClient

DATA = Path(
    "experiments/retrieval/benchmarks/"
    "hard_queries_v1.jsonl"
)

questions = [
    json.loads(line)
    for line in DATA.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

client = QdrantClient(url="http://127.0.0.1:6333")

records = []
offset = None

while True:
    batch, offset = client.scroll(
        collection_name="power_safety_children",
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    records.extend(batch)
    if offset is None:
        break

corpus = {
    f'{r.payload["parent_id"]}:{r.payload["child_index"]}':
    r.payload["text"]
    for r in records
}

for question in questions:
    print(f'\n{question["id"]}: {question["query"]}')
    for gold in question["gold_children"]:
        assert gold in corpus, f"Missing evidence: {gold}"
        print(f"  {gold}: {corpus[gold][:120]}")

print("\nAll Gold IDs exist!")
client.close()

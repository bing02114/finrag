
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from finrag.ingestion.loader import DocumentLoader
from finrag.chunking.structure_chunker import StructureChunker
from finrag.chunking.parent_child import split_article_children


MODEL_NAME = "BAAI/bge-reranker-v2-m3"

PDF_PATH = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw/pdf/"
    "D01_power_hazard_regulation.pdf"
)

# 1. Load real child chunks
pages = DocumentLoader.load_pdf(str(PDF_PATH), "D01")
articles = StructureChunker.split_articles(pages)

children = []

for article in articles:
    children.extend(split_article_children(article))

child_map = {
    f'{child["parent_id"]}:{child["child_index"]}': child["text"]
    for child in children
}

# 2. Load Cross-Encoder
print("Loading reranker:", MODEL_NAME)

reranker = CrossEncoder(
    MODEL_NAME,
    device="cuda",
    max_length=512,
    activation_fn=torch.nn.Identity(),
)

# 3. Define tests
tests = [
    {
        "id": "H08_original",
        "query": (
            "发电机组未配置电力系统稳定器，"
            "与未开展进相试验，前者属于哪一项？"
        ),
        "candidates": [
            "D01_第五条:2",
            "D01_第五条:3",
        ],
    },
    {
        "id": "H08_rewritten",
        "query": (
            "发电机组未按要求配置电力系统稳定器，"
            "属于哪一项重大事故隐患？"
        ),
        "candidates": [
            "D01_第五条:2",
            "D01_第五条:3",
        ],
    },
]

# 4. Score query-document pairs
for test in tests:
    print(f'\n===== {test["id"]} =====')

    query = test["query"]
    candidate_ids = test["candidates"]

    pairs = [
        (query, child_map[child_id])
        for child_id in candidate_ids
    ]

    start = time.perf_counter()

    scores = reranker.predict(
        pairs,
        batch_size=4,
        show_progress_bar=False,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    print("Query:", query)
    print("Latency:", round(elapsed, 4), "s")

    for rank, (child_id, score) in enumerate(ranked, 1):
        print(
            f"{rank}. {child_id} "
            f"Score={float(score):.4f}"
        )
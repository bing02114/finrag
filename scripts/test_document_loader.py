
from pathlib import Path

from finrag.ingestion.loader import DocumentLoader


BASE = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw"
)

documents = [
    (
        BASE / "pdf/D01_power_hazard_regulation.pdf",
        "D01",
        "pdf",
    ),
    (
        BASE / "html/D02_power_safety_plan.html",
        "D02",
        "html",
    ),
    (
        BASE / "html/D03_power_accident_regulation.html",
        "D03",
        "html",
    ),
]

for path, document_id, file_type in documents:

    print("\n" + "=" * 60)
    print("Document:", document_id)

    if file_type == "pdf":
        pages = DocumentLoader.load_pdf(
            str(path),
            document_id,
        )
    else:
        pages = DocumentLoader.load_html(
            str(path),
            document_id,
        )

    print("Records:", len(pages))

    total_chars = sum(
        len(page["text"])
        for page in pages
    )

    print("Characters:", total_chars)

    for page in pages[:2]:
        print("\nPage:", page["page_number"])
        print(page["text"][:300])

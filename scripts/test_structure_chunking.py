
from pathlib import Path

from finrag.ingestion.loader import DocumentLoader
from finrag.chunking.structure_chunker import StructureChunker


PDF_PATH = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw/pdf/"
    "D01_power_hazard_regulation.pdf"
)

# 1. Load all PDF pages
pages = DocumentLoader.load_pdf(
    str(PDF_PATH),
    document_id="D01",
)

# 2. Structure-aware chunking
chunks = StructureChunker.split_articles(pages)

# 3. Show summary
print("===== STRUCTURE CHUNKING =====")
print("PDF pages:", len(pages))
print("Logical chunks:", len(chunks))

for chunk in chunks:
    print(
        chunk["article"],
        "|",
        chunk["chapter"],
        "| Pages:",
        chunk["start_page"],
        "-",
        chunk["end_page"],
        "| Length:",
        len(chunk["text"]),
    )

# 4. Check article five
print("\n===== ARTICLE FIVE =====")

article_five = next(
    c for c in chunks
    if c["article"] == "第五条"
)

print("Chapter:", article_five["chapter"])
print("Start page:", article_five["start_page"])
print("End page:", article_five["end_page"])
print("Text preview:")
print(article_five["text"][:600])

# 5. Verify cross-page assembly
assert article_five["start_page"] == 1
assert article_five["end_page"] >= 2

assert "判定为一项重大隐患" in article_five["text"]

print("\nCross-page assembly test passed!")

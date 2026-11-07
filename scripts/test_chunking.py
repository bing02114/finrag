
from pathlib import Path

from finrag.ingestion.loader import DocumentLoader
from finrag.ingestion.cleaner import clean_pdf_text
from finrag.chunking.chunker import TextChunker


pdf_path = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw/pdf/"
    "D01_power_hazard_regulation.pdf"
)

pages = DocumentLoader.load_pdf(
    str(pdf_path),
    "D01",
)

# 先使用第一页做对比
text = clean_pdf_text(pages[0]["text"])

fixed_chunks = TextChunker.fixed_size(
    text,
    chunk_size=200,
    overlap=30,
)

recursive_chunks = TextChunker.recursive(
    text,
    chunk_size=200,
    overlap=30,
)

print("===== FIXED SIZE =====")
print("Chunks:", len(fixed_chunks))

for i, chunk in enumerate(fixed_chunks):
    print(f"\n[{i}] Length: {len(chunk)}")
    print(chunk)


print("\n===== RECURSIVE =====")
print("Chunks:", len(recursive_chunks))

for i, chunk in enumerate(recursive_chunks):
    print(f"\n[{i}] Length: {len(chunk)}")
    print(chunk)

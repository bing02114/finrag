uv run --group documents python - <<'PY'
from pathlib import Path
import fitz
from bs4 import BeautifulSoup

base = Path("/workspace/finrag-artifacts/enterprise-kb/raw")

pdf_path = base / "pdf/D01_power_hazard_regulation.pdf"

with fitz.open(pdf_path) as doc:
    print("PDF pages:", len(doc))
    print("First page preview:")
    print(doc[0].get_text()[:400])

for path in sorted((base / "html").glob("*.html")):
    soup = BeautifulSoup(
        path.read_bytes(),
        "html.parser"
    )

    print("\nHTML:", path.name)
    print("Title:", soup.title.get_text(" ", strip=True) if soup.title else "N/A")
    print("Text preview:", soup.get_text(" ", strip=True)[:300])
PY

from pathlib import Path

from finrag.ingestion.loader import DocumentLoader
from finrag.ingestion.cleaner import (
    clean_pdf_text,
    clean_html_text,
)

BASE = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw"
)

# 1. PDF Cleaning
pdf_path = BASE / "pdf/D01_power_hazard_regulation.pdf"

pages = DocumentLoader.load_pdf(
    str(pdf_path),
    "D01",
)

print("===== PDF CLEANING =====")

original = pages[0]["text"]
cleaned = clean_pdf_text(original)

print("Original length:", len(original))
print("Cleaned length:", len(cleaned))

print("\nOriginal:")
print(original[:500])

print("\nCleaned:")
print(cleaned[:500])


# 2. HTML Cleaning
print("\n===== HTML CLEANING =====")

for name in [
    "D02_power_safety_plan.html",
    "D03_power_accident_regulation.html",
]:

    path = BASE / "html" / name

    original_html = path.read_bytes()
    cleaned = clean_html_text(original_html)

    print("\nFile:", name)
    print("Cleaned length:", len(cleaned))
    print("Preview:", cleaned[:500])



from bs4 import BeautifulSoup
from pathlib import Path

path = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw/html/"
    "D03_power_accident_regulation.html"
)

soup = BeautifulSoup(
    path.read_bytes(),
    "html.parser"
)

keyword = "第一章"

for tag in soup.find_all(string=True):
    if keyword in tag:
        parent = tag.parent

        print("Tag:", parent.name)
        print("Class:", parent.get("class"))
        print("ID:", parent.get("id"))

        for ancestor in list(parent.parents)[:5]:
            print(
                ancestor.name,
                ancestor.get("class"),
                ancestor.get("id")
            )
        break
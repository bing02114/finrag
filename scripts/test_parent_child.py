
from pathlib import Path

from finrag.ingestion.loader import DocumentLoader
from finrag.chunking.structure_chunker import StructureChunker
from finrag.chunking.parent_child import split_article_children


pdf_path = Path(
    "/workspace/finrag-artifacts/enterprise-kb/raw/pdf/"
    "D01_power_hazard_regulation.pdf"
)

pages = DocumentLoader.load_pdf(str(pdf_path), "D01")
articles = StructureChunker.split_articles(pages)

article_five = next(
    a for a in articles if a["article"] == "第五条"
)

children = split_article_children(article_five)

print("===== PARENT =====")
print("Article:", article_five["article"])
print("Length:", len(article_five["text"]))

print("\n===== CHILDREN =====")
print("Child count:", len(children))

for child in children:
    print(
        f'\nChild {child["child_index"]}',
        f'Length: {len(child["text"])}',
        f'Parent: {child["parent_id"]}',
    )
    print(child["text"][:250])

assert len(children) > 1
assert all(c["parent_id"] == "D01_第五条" for c in children)

print("\nParent-Child test passed!")


from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup


class DocumentLoader:

    @staticmethod
    def load_pdf(file_path: str, document_id: str):
        pages = []

        with pymupdf.open(file_path) as doc:
            for page_index, page in enumerate(doc):

                text = page.get_text("text")

                pages.append({
                    "document_id": document_id,
                    "source_file": Path(file_path).name,
                    "page_number": page_index + 1,
                    "text": text,
                })

        return pages

    @staticmethod
    def load_html(file_path: str, document_id: str):

        with open(file_path, "rb") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Remove non-content elements
        for element in soup(
            ["script", "style", "nav", "footer"]
        ):
            element.decompose()

        # Extract text
        text = soup.get_text(
            separator="\n",
            strip=True
        )

        return [{
            "document_id": document_id,
            "source_file": Path(file_path).name,
            "page_number": None,
            "text": text,
        }]

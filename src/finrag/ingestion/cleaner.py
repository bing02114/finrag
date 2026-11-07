
import re
from bs4 import BeautifulSoup



import re


def clean_pdf_text(text: str) -> str:
    # 1. Normalize whitespace
    text = text.replace("\r\n", "\n")
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")

    # 2. Merge visual line breaks
    text = re.sub(r"\s*\n\s*", "", text)

    # 3. Restore chapter boundaries
    text = re.sub(
        r"(第[一二三四五六七八九十百零〇两\d]+章)",
        r"\n\n\1 ",
        text
    )

    # 4. Restore article boundaries
    text = re.sub(
        r"(第[一二三四五六七八九十百零〇两\d]+条)",
        r"\n\n\1 ",
        text
    )

    # 5. Restore numbered item boundaries
    text = re.sub(
        r"([（(][一二三四五六七八九十百零〇两\d]+[）)])",
        r"\n\1",
        text
    )

    # 6. Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_html_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(
        ["script", "style", "nav", "footer"]
    ):
        tag.decompose()

    # Locate main article content
    content = (
        soup.select_one("#detailContent")
        or soup.select_one(".article-content")
        or soup.find("article")
        or soup.find("main")
    )

    if content is None:
        raise ValueError(
            "Unable to locate the main HTML content"
        )

    text = content.get_text(
        separator="\n",
        strip=True
    )

    # Normalize whitespace
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

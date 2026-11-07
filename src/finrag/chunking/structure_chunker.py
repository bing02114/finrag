
import re
from typing import Optional


# Match chapter headings at the start of a line
CHAPTER_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百零〇两\d]+章)\s*(.*)"
)

# Match article headings at the start of a line
ARTICLE_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百零〇两\d]+条)\s*(.*)"
)

# Match numbered items such as （一）, （二）
ITEM_PATTERN = re.compile(
    r"^[（(][一二三四五六七八九十百零〇两\d]+[）)]"
)


class StructureChunker:

    @staticmethod
    def split_articles(pages: list[dict]) -> list[dict]:
        """
        Convert PDF page records into article-level chunks.

        Input:
            document_id, page_number, text

        Output:
            article, chapter, start_page,
            end_page, text
        """

        chunks = []
        current_chapter = None
        current_article: Optional[dict] = None

        def save_article():
            nonlocal current_article

            if current_article is not None:
                current_article["text"] = "\n".join(
                    current_article.pop("parts")
                ).strip()

                chunks.append(current_article)
                current_article = None

        for page in pages:
            page_number = page["page_number"]

            # Use the original extracted lines.
            # Do not clean each page separately.
            text = (
                page["text"]
                .replace("\r\n", "\n")
                .replace("\u3000", " ")
                .replace("\xa0", " ")
            )

            for raw_line in text.splitlines():
                line = raw_line.strip()

                if not line:
                    continue

                # 1. Detect new chapter
                chapter_match = CHAPTER_PATTERN.match(line)

                if chapter_match:
                    save_article()

                    number = chapter_match.group(1)
                    title = chapter_match.group(2)

                    current_chapter = f"{number} {title}".strip()
                    continue

                # 2. Detect new article
                article_match = ARTICLE_PATTERN.match(line)

                if article_match:
                    save_article()

                    article_number = article_match.group(1)

                    current_article = {
                        "document_id": page["document_id"],
                        "chapter": current_chapter,
                        "article": article_number,
                        "start_page": page_number,
                        "end_page": page_number,
                        "parts": [line],
                    }
                    continue

                # 3. Append content to current article
                if current_article is not None:
                    current_article["end_page"] = page_number

                    if ITEM_PATTERN.match(line):
                        # Keep numbered items separate
                        current_article["parts"].append(line)
                    else:
                        # Merge PDF visual line breaks
                        current_article["parts"][-1] += line

        # Save the final article
        save_article()

        return chunks

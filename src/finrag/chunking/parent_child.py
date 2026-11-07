
import re

ITEM_START = re.compile(
    r"(?=（[一二三四五六七八九十百零〇两\d]+）)"
)


def split_article_children(article: dict) -> list[dict]:
    text = article["text"]

    # Split before numbered items
    parts = ITEM_START.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    parent_id = (
        f'{article["document_id"]}_{article["article"]}'
    )

    # The first part is the article introduction
    has_items = len(parts) > 1 and bool(
        re.match(r"^（[一二三四五六七八九十百零〇两\d]+）", parts[1])
    )

    if has_items:
        introduction = parts[0]
        items = parts[1:]
    else:
        introduction = ""
        items = [text]

    children = []

    for i, item in enumerate(items):
        # Preserve context in the embedding text
        context = (
            f'{article["chapter"] or ""}\n'
            f'{introduction}\n'
            if introduction else
            f'{article["chapter"] or ""}\n'
        )

        children.append({
            "parent_id": parent_id,
            "child_index": i,
            "document_id": article["document_id"],
            "article": article["article"],
            "chapter": article["chapter"],
            "start_page": article["start_page"],
            "end_page": article["end_page"],
            "text": (context + item).strip(),
        })

    return children

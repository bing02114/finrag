
import os
import json
import argparse
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field


MODEL = "deepseek-chat"

DATA_PATH = Path(
    "experiments/retrieval/benchmarks/"
    "hard_queries_v1.jsonl"
)

OUTPUT_PATH = Path(
    "experiments/retrieval/"
    "query_rewrite_v1.json"
)


class RewriteResult(BaseModel):
    rewritten_query: str
    changed: bool


SYSTEM_PROMPT = """
你是企业知识库 RAG 系统的 Query Rewrite 模块。

你的任务是把用户的问题改写成清晰、准确、
适合知识库检索的中文问题。

要求：

1. 保留用户真正想询问的内容。
2. 解析前者、后者、它等指代关系。
3. 多个独立问题必须全部保留。
4. 保留设备类型、数量、电压等级和否定条件。
5. 不要添加原文没有支持的事实。
6. 不要擅自将两个专业术语认定为等价。
7. 不要回答用户的问题，只改写问题。
8. 如果原问题已经清晰，可以保持原文。

必须输出合法 JSON，格式如下：

{
    "rewritten_query": "...",
    "changed": true
}
"""


def rewrite_query(client, query):

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0,
        max_tokens=500,
        stream=False,
    )

    elapsed = time.perf_counter() - start

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("DeepSeek returned empty JSON")

    result = RewriteResult.model_validate(
        json.loads(content)
    )

    rewritten = result.rewritten_query.strip()

    if not rewritten:
        raise ValueError("Empty rewritten query")

    return {
        "rewritten_query": rewritten,
        "changed": rewritten != query.strip(),
        "latency": round(elapsed, 4),
        "usage": (
            response.usage.model_dump()
            if response.usage else None
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
    )

    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Please set DEEPSEEK_API_KEY"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    questions = [
        json.loads(line)
        for line in DATA_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    selected = (
        questions
        if args.all
        else [q for q in questions if q["id"] == "H08"]
    )

    results = []

    for question in selected:

        print(f'\n===== {question["id"]} =====')

        original = question["query"]

        result = rewrite_query(
            client,
            original,
        )

        row = {
            "id": question["id"],
            "category": question["category"],
            "original_query": original,
            **result,
        }

        results.append(row)

        print("Original:", original)
        print("Rewritten:", result["rewritten_query"])
        print("Latency:", result["latency"])
        print("Usage:", result["usage"])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()


import os
import json
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
    "query_decomposition_v1.json"
)


# =====================================
# 1. Structured Output
# =====================================

class DecompositionResult(BaseModel):
    queries: list[str] = Field(
        description="拆分后的独立检索问题"
    )


# =====================================
# 2. System Prompt
# =====================================

SYSTEM_PROMPT = """
你是企业知识库 RAG 系统中的 Query Decomposition 模块。

你的任务是分析用户的问题，并将包含多个独立信息需求的
复合问题拆分成多个可以独立检索的子问题。

规则：

1. 每个独立的信息需求对应一个子问题。
2. 每个子问题必须能够脱离原问题独立理解。
3. 保留原始问题中的设备、数值、电压等级、
   否定条件和其他重要约束。
4. 不得添加原问题没有提供的事实。
5. 不得自行补充法规内容或判定结论。
6. 不要把两个不同的判定条件合并成一个问题。
7. 不要过度拆分，一个完整的信息需求只生成一个子问题。
8. 如果原问题只有一个独立信息需求，
   则只返回一个子问题。
9. 不要回答问题，只进行分解。
10. 输出必须是合法 JSON。

输出格式：

{
    "queries": [
        "子问题1",
        "子问题2"
    ]
}
"""


# =====================================
# 3. DeepSeek Client
# =====================================

def create_client():

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured"
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


# =====================================
# 4. Decomposition
# =====================================

def decompose_query(client, query):

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

    if not content:
        raise RuntimeError(
            "Empty response from DeepSeek"
        )

    result = DecompositionResult.model_validate_json(
        content
    )

    queries = [
        q.strip()
        for q in result.queries
        if q.strip()
    ]

    # Remove duplicates while preserving order
    queries = list(dict.fromkeys(queries))

    if not queries:
        raise ValueError(
            "Decomposition returned no queries"
        )

    return {
        "queries": queries,
        "latency": round(elapsed, 4),
        "usage": (
            response.usage.model_dump()
            if response.usage else None
        ),
    }


# =====================================
# 5. Load Multi-evidence Questions
# =====================================

questions = [
    json.loads(line)
    for line in DATA_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

selected = [
    q for q in questions
    if q["category"] == "multi_evidence"
]


# =====================================
# 6. Run Experiments
# =====================================

def main():

    client = create_client()

    results = []

    for question in selected:

        qid = question["id"]
        original = question["query"]

        print(f"\n===== {qid} =====")
        print("Original:", original)

        result = decompose_query(
            client,
            original,
        )

        row = {
            "id": qid,
            "category": question["category"],
            "original_query": original,
            **result,
        }

        results.append(row)

        print(
            "Sub-query count:",
            len(result["queries"])
        )

        for i, query in enumerate(
            result["queries"], 1
        ):
            print(f"Query {i}: {query}")

        print(
            "Latency:",
            result["latency"],
            "s"
        )

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

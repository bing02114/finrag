
import time

from pydantic import BaseModel, Field

from finrag.llm.client import (
    create_llm_client,
    get_llm_model,
)


# ==========================================
# 1. Structured Output
# ==========================================

class DecompositionResult(BaseModel):

    queries: list[str] = Field(
        description="拆分后的独立检索问题"
    )


# ==========================================
# 2. System Prompt
# ==========================================

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


# ==========================================
# 3. Query Decomposer
# ==========================================

class QueryDecomposer:

    def __init__(self, client=None, model=None):

        self.client = client or create_llm_client()

        self.model = model or get_llm_model()

    def decompose(
        self,
        query: str,
    ) -> dict:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"原始用户问题：\n{query}\n\n"
                        "请将其拆分为独立检索问题。"
                    ),
                },
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0,
            max_tokens=800,
            stream=False,
        )

        elapsed = time.perf_counter() - start

        if not response.choices:
            raise RuntimeError(
                "Decomposer returned no choices."
            )

        choice = response.choices[0]

        content = choice.message.content

        if not content or not content.strip():

            print(
                "Decomposer finish reason:",
                choice.finish_reason,
            )

            print(
                "Decomposer refusal:",
                choice.message.refusal,
            )

            raise RuntimeError(
                "Decomposer returned empty content."
            )

        result = DecompositionResult.model_validate_json(
            content
        )

        # Remove empty queries
        queries = [
            item.strip()
            for item in result.queries
            if item and item.strip()
        ]

        # Remove duplicates while preserving order
        queries = list(
            dict.fromkeys(queries)
        )

        if not queries:
            raise ValueError(
                "Decomposition returned no queries."
            )

        return {
            "queries": queries,
            "latency": round(elapsed, 4),
            "usage": (
                response.usage.model_dump()
                if response.usage
                else None
            ),
        }


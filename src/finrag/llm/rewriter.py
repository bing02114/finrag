
from finrag.llm.client import (
    create_llm_client,
    get_llm_model,
)


SYSTEM_PROMPT = """
你是企业知识库 RAG 系统的 Query Rewrite 模块。

根据用户原始问题和当前缺失的信息，
生成一个针对缺失内容的独立检索问题。

要求：

1. 只针对尚未获得证据支持的信息。
2. 保留原始问题中的设备、数值和适用条件。
3. 不添加原问题没有提供的事实。
4. 不要回答问题。
5. 生成一个清晰、独立、适合知识库检索的中文问题。
6. 不要重复已经获得证据支持的信息。

只输出检索问题，不要解释。
"""


class QueryRewriter:

    def __init__(self, client=None, model=None):
    
        self.client = client or create_llm_client()
        self.model = model or get_llm_model()

    def rewrite(
        self,
        original_query: str,
        missing_aspects: list[str],
    ) -> str:

        if not missing_aspects:
            raise ValueError(
                "Missing aspects cannot be empty."
            )

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
                        f"原始问题：{original_query}\n\n"
                        f"已经缺失的信息：\n"
                        + "\n".join(
                            f"- {item}"
                            for item in missing_aspects
                        )
                    ),
                },
            ],
            temperature=0,
            max_tokens=300,
        )

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise RuntimeError(
                "Query Rewrite returned empty content."
            )

        return content.strip()

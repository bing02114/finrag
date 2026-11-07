
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": "你是一个专业的中文 RAG Query Rewrite 助手。"
        },
        {
            "role": "user",
            "content": (
                "发电机组未配置电力系统稳定器，"
                "与未开展进相试验，前者属于哪一项？"
            )
        }
    ],
    stream=False,
)

print(response.choices[0].message.content)
print("Model:", response.model)
print("Usage:", response.usage)

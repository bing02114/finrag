
import os

from openai import OpenAI


def create_llm_client() -> OpenAI:
    """
    Create a DeepSeek-compatible OpenAI client.
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
        timeout=60.0,
        max_retries=2,
    )


def get_llm_model() -> str:
    """
    Read model name from environment.
    """

    return os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-flash",
    )

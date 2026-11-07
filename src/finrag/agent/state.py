
from typing import Annotated, TypedDict


# ==========================================
# 1. Reducers
# ==========================================

def merge_unique(
    existing: list[str],
    new: list[str],
) -> list[str]:

    return list(dict.fromkeys(existing + new))


def merge_evidence(
    existing: list[dict],
    new: list[dict],
) -> list[dict]:

    merged = {
        item["id"]: item
        for item in existing
    }

    for item in new:
        merged[item["id"]] = item

    return list(merged.values())


def append_history(
    existing: list[dict],
    new: list[dict],
) -> list[dict]:

    return existing + new


def merge_query_results(
    existing: list[dict],
    new: list[dict],
) -> list[dict]:
    """
    Merge parallel retrieval results by query.

    If the same query is executed again,
    its latest result replaces the old one.
    """

    merged = {
        item["query"]: item
        for item in existing
    }

    for item in new:
        merged[item["query"]] = item

    return list(merged.values())


# ==========================================
# 2. RAG State
# ==========================================

class RAGState(TypedDict):

    # ======================================
    # User Input
    # ======================================

    original_query: str

    conversation_context: str

    current_query: str

    # ======================================
    # Query Analysis
    # ======================================

    query_strategy: str

    retrieval_query: str

    clarification_question: str

    clarification_response: str

    # ======================================
    # Query Decomposition
    # ======================================

    sub_queries: list[str]

    worker_query: str

    # ======================================
    # Parallel Retrieval Results
    # ======================================

    query_results: Annotated[
        list[dict],
        merge_query_results,
    ]

    # ======================================
    # Accumulated Evidence
    # ======================================

    evidence_ids: Annotated[
        list[str],
        merge_unique,
    ]

    evidence: Annotated[
        list[dict],
        merge_evidence,
    ]

    # ======================================
    # Grading
    # ======================================

    grading_status: str

    covered_aspects: list[str]

    missing_aspects: list[str]

    # ======================================
    # Retry Management
    # ======================================

    retry_count: int

    max_retries: int

    # ======================================
    # Query Tracking
    # ======================================

    seen_queries: Annotated[
        list[str],
        merge_unique,
    ]

    # ======================================
    # Execution History
    # ======================================

    history: Annotated[
        list[dict],
        append_history,
    ]

    # ======================================
    # Final Result
    # ======================================

    final_answer: str

    failure_reason: str

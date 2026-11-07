
from langgraph.types import Send

from finrag.agent.state import RAGState


# ==========================================
# 1. Route After Query Analysis
# ==========================================

def route_after_analysis(state: RAGState):

    strategy = state["query_strategy"]

    if strategy == "direct":
        return "retrieve"

    if strategy == "decompose":
        return "decompose"

    if strategy == "clarify":
        return "clarify"

    return "fallback"


# ==========================================
# 2. Dispatch Parallel Queries
# ==========================================

def dispatch_subqueries(state: RAGState):

    queries = state["sub_queries"]

    if not queries:
        raise ValueError(
            "No sub-queries available for dispatch."
        )

    print("\n[Send API] Dispatching sub-queries")

    print("Task count:", len(queries))

    return [
        Send(
            "retrieve_subquery",
            {
                "worker_query": query,
            },
        )
        for query in queries
    ]


# ==========================================
# 3. Route After Clarification
# ==========================================

def route_after_clarify(state: RAGState):

    if (
        state["failure_reason"]
        or not state["clarification_response"]
    ):
        return "fallback"

    return "retrieve"


# ==========================================
# 4. Route After Grading
# ==========================================

def route_after_grading(state: RAGState):

    if state["grading_status"] == "sufficient":
        return "answer"

    if state["retry_count"] >= state["max_retries"]:
        return "fallback"

    if not state["missing_aspects"]:
        return "fallback"

    return "rewrite"


# ==========================================
# 5. Route After Rewrite
# ==========================================

def route_after_rewrite(state: RAGState):

    if state["failure_reason"]:
        return "fallback"

    return "retrieve"

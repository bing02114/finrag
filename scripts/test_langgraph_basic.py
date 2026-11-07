
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ==========================================
# 1. Define State
# ==========================================

class RAGState(TypedDict):

    original_query: str

    current_query: str

    evidence_ids: list[str]

    grading_status: str

    missing_aspects: list[str]

    retry_count: int

    final_answer: str


# ==========================================
# 2. Retrieval Node
# ==========================================

def retrieval_node(state: RAGState):

    print("\n[Retrieval Node]")
    print("Query:", state["current_query"])

    # Simulate first retrieval
    if state["retry_count"] == 0:

        new_evidence = [
            "D01_第五条:5"
        ]

    # Simulate targeted retrieval
    else:

        new_evidence = [
            "D01_第五条:6"
        ]

    # Deduplicate
    evidence = list(
        dict.fromkeys(
            state["evidence_ids"] + new_evidence
        )
    )

    print("Retrieved:", new_evidence)

    return {
        "evidence_ids": evidence
    }


# ==========================================
# 3. Grading Node
# ==========================================

def grading_node(state: RAGState):

    print("\n[Grading Node]")

    evidence = set(state["evidence_ids"])

    gold = {
        "D01_第五条:5",
        "D01_第五条:6",
    }

    covered = evidence.intersection(gold)
    missing = gold - evidence

    if not missing:

        status = "sufficient"

    elif covered:

        status = "partial"

    else:

        status = "insufficient"

    print("Status:", status)
    print("Covered:", covered)
    print("Missing:", missing)

    return {
        "grading_status": status,
        "missing_aspects": sorted(missing)
    }


# ==========================================
# 4. Rewrite Node
# ==========================================

def rewrite_node(state: RAGState):

    print("\n[Rewrite Node]")

    # Simulate generating a targeted query
    followup_query = (
        "变压器基础沉降差或倾斜"
        "超过什么标准属于重大隐患？"
    )

    print("Follow-up Query:", followup_query)

    return {
        "current_query": followup_query,
        "retry_count": state["retry_count"] + 1
    }


# ==========================================
# 5. Answer Node
# ==========================================

def answer_node(state: RAGState):

    print("\n[Answer Node]")

    answer = (
        "已经找到乙炔含量和基础沉降两项"
        "重大隐患判定条件的证据。"
    )

    print("Answer:", answer)

    return {
        "final_answer": answer
    }


# ==========================================
# 6. Fallback Node
# ==========================================

def fallback_node(state: RAGState):

    print("\n[Fallback Node]")

    answer = (
        "当前检索证据不足，"
        "无法可靠回答全部问题。"
    )

    return {
        "final_answer": answer
    }


# ==========================================
# 7. Conditional Routing
# ==========================================

def route_after_grading(state: RAGState):

    if state["grading_status"] == "sufficient":

        return "answer"

    if state["retry_count"] >= 2:

        return "fallback"

    return "rewrite"


# ==========================================
# 8. Build Graph
# ==========================================

builder = StateGraph(RAGState)

# Register Nodes
builder.add_node(
    "retrieve",
    retrieval_node
)

builder.add_node(
    "grade",
    grading_node
)

builder.add_node(
    "rewrite",
    rewrite_node
)

builder.add_node(
    "answer",
    answer_node
)

builder.add_node(
    "fallback",
    fallback_node
)


# Register Edges
builder.add_edge(START, "retrieve")

builder.add_edge(
    "retrieve",
    "grade"
)

builder.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "answer": "answer",
        "rewrite": "rewrite",
        "fallback": "fallback",
    }
)

builder.add_edge(
    "rewrite",
    "retrieve"
)

builder.add_edge("answer", END)

builder.add_edge("fallback", END)


# Compile
graph = builder.compile()


# ==========================================
# 9. Invoke Graph
# ==========================================

initial_state: RAGState = {

    "original_query": (
        "变压器乙炔超标和基础沉降超标"
        "分别对应什么重大隐患条件？"
    ),

    "current_query": (
        "变压器乙炔超标和基础沉降超标"
        "分别对应什么重大隐患条件？"
    ),

    "evidence_ids": [],

    "grading_status": "",

    "missing_aspects": [],

    "retry_count": 0,

    "final_answer": "",
}


print("\n========== START GRAPH ==========")

result = graph.invoke(initial_state)

print("\n========== FINAL STATE ==========")

print("Status:", result["grading_status"])

print("Evidence:", result["evidence_ids"])

print("Retries:", result["retry_count"])

print("Answer:", result["final_answer"])

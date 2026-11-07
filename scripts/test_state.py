
from langgraph.graph import StateGraph, START, END

from finrag.agent.state import RAGState


def first_retrieval(state: RAGState):

    return {
        "evidence_ids": [
            "D01_第五条:5",
            "D01_第五条:6",
        ],
        "history": [
            {"step": 1, "node": "first_retrieval"}
        ],
    }


def second_retrieval(state: RAGState):

    return {
        "evidence_ids": [
            "D01_第五条:6",
            "D01_第六条:0",
        ],
        "history": [
            {"step": 2, "node": "second_retrieval"}
        ],
    }


builder = StateGraph(RAGState)

builder.add_node("first", first_retrieval)
builder.add_node("second", second_retrieval)

builder.add_edge(START, "first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

graph = builder.compile()


initial_state: RAGState = {
    "original_query": "测试问题",
    "current_query": "测试问题",
    "evidence_ids": [],
    "grading_status": "",
    "covered_aspects": [],
    "missing_aspects": [],
    "retry_count": 0,
    "max_retries": 2,
    "seen_queries": ["测试问题"],
    "history": [],
    "final_answer": "",
    "failure_reason": "",
}


result = graph.invoke(initial_state)


print("===== STATE REDUCER TEST =====")

print("Evidence IDs:")
for item in result["evidence_ids"]:
    print(item)

print("\nHistory:")
for item in result["history"]:
    print(item)

print("\nEvidence count:", len(result["evidence_ids"]))
print("History count:", len(result["history"]))

assert len(result["evidence_ids"]) == 3
assert len(result["history"]) == 2

print("\nState Reducer test passed!")

from finrag.agent.routing import route_after_clarify
from langgraph.graph import StateGraph, START, END

from finrag.agent.state import RAGState

from finrag.agent.nodes import create_nodes

from finrag.agent.routing import (
    route_after_analysis,
    dispatch_subqueries,
    route_after_clarify,
    route_after_grading,
    route_after_rewrite,
)

from finrag.retrieval.pipeline import HybridRetriever

from finrag.llm.client import (
    create_llm_client,
    get_llm_model,
)

from finrag.llm.grader import EvidenceGrader

from finrag.llm.rewriter import QueryRewriter

from finrag.llm.analyzer import QueryAnalyzer

from finrag.llm.decomposer import QueryDecomposer


def create_agent(checkpointer=None):
    """
    Create FinRAG Agent v2.

    Features:
        - Query Analyzer
        - Query Decomposition
        - LangGraph Send API
        - Parallel Retrieval
        - RRF Evidence Fusion
        - Adaptive Grading and Rewriting
        - Checkpoint Persistence
    """

    print(
        "\n========== INITIALIZING FINRAG AGENT V2 =========="
    )

    # ======================================
    # 1. Initialize Retriever
    # ======================================

    retriever = HybridRetriever(
        collection_name="power_safety_children",
        device="cuda",
        recall_k=20,
        candidate_k=20,
        final_k=10,
    )

    # ======================================
    # 2. Initialize Shared LLM Client
    # ======================================

    llm_client = create_llm_client()

    llm_model = get_llm_model()

    # ======================================
    # 3. Initialize LLM Modules
    # ======================================

    analyzer = QueryAnalyzer(
        client=llm_client,
        model=llm_model,
    )

    decomposer = QueryDecomposer(
        client=llm_client,
        model=llm_model,
    )

    grader = EvidenceGrader(
        client=llm_client,
        model=llm_model,
    )

    rewriter = QueryRewriter(
        client=llm_client,
        model=llm_model,
    )

    # ======================================
    # 4. Create Nodes
    # ======================================

    nodes = create_nodes(
        retriever=retriever,
        grader=grader,
        rewriter=rewriter,
        analyzer=analyzer,
        decomposer=decomposer,
        llm_client=llm_client,
        llm_model=llm_model,
        max_retrieval_concurrency=2,
    )

    # ======================================
    # 5. Validate Nodes
    # ======================================

    required_nodes = {
        "analyze",
        "decompose",
        "retrieve",
        "retrieve_subquery",
        "merge_subquery",
        "clarify",
        "grade",
        "rewrite",
        "answer",
        "fallback",
    }

    missing_nodes = (
        required_nodes - set(nodes.keys())
    )

    if missing_nodes:

        raise ValueError(
            f"Missing Graph Nodes: {missing_nodes}"
        )

    # ======================================
    # 6. Build StateGraph
    # ======================================

    builder = StateGraph(RAGState)

    for name, node in nodes.items():

        builder.add_node(
            name,
            node,
        )

    # ======================================
    # 7. Start
    # ======================================

    builder.add_edge(
        START,
        "analyze",
    )

    # ======================================
    # 8. Analyzer Routing
    # ======================================

    builder.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "decompose": "decompose",
            "clarify": "clarify",
            "fallback": "fallback",
        },
    )

    # ======================================
    # 9. Direct Retrieval
    # ======================================

    builder.add_edge(
        "retrieve",
        "grade",
    )

    # ======================================
    # 10. Parallel Retrieval
    # ======================================

    builder.add_conditional_edges(
        "decompose",
        dispatch_subqueries,
    )

    # All parallel workers must finish
    # before merge_subquery is executed.

    builder.add_edge(
        "retrieve_subquery",
        "merge_subquery",
    )

    builder.add_edge(
        "merge_subquery",
        "grade",
    )

    # ======================================
    # 11. Clarification
    # ======================================

    builder.add_conditional_edges(
        "clarify",
        route_after_clarify,
        {
            "retrieve": "retrieve",
            "fallback": "fallback",
        },
    )

    # ======================================
    # 12. Grading Routing
    # ======================================

    builder.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "answer": "answer",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )

    # ======================================
    # 13. Rewrite Routing
    # ======================================

    builder.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {
            "retrieve": "retrieve",
            "fallback": "fallback",
        },
    )

    # ======================================
    # 14. End Nodes
    # ======================================

    builder.add_edge(
        "answer",
        END,
    )

    builder.add_edge(
        "fallback",
        END,
    )

    # ======================================
    # 15. Compile
    # ======================================

    graph = builder.compile(
        checkpointer=checkpointer
    )

    print("\nFinRAG Agent v2 compiled successfully.")

    print("LLM Model:", llm_model)

    print(
        "Checkpointer:",
        type(checkpointer).__name__
        if checkpointer
        else "Disabled",
    )

    print("Retrieval Concurrency: 2")

    return graph, retriever, llm_client

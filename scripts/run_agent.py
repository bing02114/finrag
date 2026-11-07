import argparse
import json

import sqlite3

from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

from finrag.agent.graph import create_agent


def main():

    # ======================================
    # 1. Parse Arguments
    # ======================================

    parser = argparse.ArgumentParser(
        description="Run FinRAG Agent"
    )

    parser.add_argument(
        "--query",
        type=str,
        default=(
            "变压器乙炔超标和基础沉降超标"
            "分别对应什么重大隐患条件？"
        ),
    )

    parser.add_argument(
        "--thread-id",
        type=str,
        default="finrag-demo-h10",
    )

    args = parser.parse_args()

    query = args.query
    thread_id = args.thread_id

    # ======================================
    # 2. Create Checkpointer
    # ======================================

    DB_PATH = Path(
    "data/checkpoints/finrag.db"
    )
    
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=30,
    )
    
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    # ======================================
    # 3. Create Agent
    # ======================================

    graph, retriever, llm_client = create_agent(
        checkpointer=checkpointer
    )

    # ======================================
    # 4. Configure Thread
    # ======================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 20,
    }

    # ======================================
    # 5. Initial State
    # ======================================

    initial_state = {
    
        # User Input
        "original_query": query,
        "conversation_context": "",
        "current_query": query,
    
        # Query Analysis
        "query_strategy": "",
        "retrieval_query": "",
        "clarification_question": "",
        "clarification_response": "",
    
        # Decomposition
        "sub_queries": [],
        "worker_query": "",
    
        # Parallel Results
        "query_results": [],
    
        # Evidence
        "evidence_ids": [],
        "evidence": [],
    
        # Grading
        "grading_status": "",
        "covered_aspects": [],
        "missing_aspects": [],
    
        # Retry
        "retry_count": 0,
        "max_retries": 2,
    
        # Query Tracking
        "seen_queries": [query],
    
        # History
        "history": [],
    
        # Final Result
        "final_answer": "",
        "failure_reason": "",
    }

    try:

        # ==================================
        # 6. Run Agent
        # ==================================

        print("\n========== FINRAG AGENT START ==========")

        print("Thread ID:", thread_id)

        print("Query:", query)

        result = graph.invoke(
            initial_state,
            config=config,
        )

        # ==================================
        # 7. Final Result
        # ==================================

        print("\n========== FINAL RESULT ==========")

        print(
            "Original Query:",
            result["original_query"],
        )

        print(
            "Status:",
            result["grading_status"],
        )

        print(
            "Retries:",
            result["retry_count"],
        )

        print(
            "Evidence IDs:",
            result["evidence_ids"],
        )

        print(
            "Failure:",
            result["failure_reason"],
        )

        print("\n===== FINAL ANSWER =====")

        print(result["final_answer"])

        # ==================================
        # 8. Checkpoint Inspection
        # ==================================

        print("\n===== CHECKPOINT STATE =====")

        snapshot = graph.get_state(config)

        print("Values:")

        print(
            json.dumps(
                snapshot.values,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        print("\nNext:")

        print(snapshot.next)

        checkpoint_id = (
            snapshot.config["configurable"]
            .get("checkpoint_id")
        )

        print("\nCheckpoint ID:")
        print(checkpoint_id)

        # ==================================
        # 9. Checkpoint History
        # ==================================

        print("\n===== CHECKPOINT HISTORY =====")

        for index, item in enumerate(
            graph.get_state_history(config),
            1,
        ):

            print(f"\nCheckpoint {index}")

            print(
                "ID:",
                item.config["configurable"].get(
                    "checkpoint_id"
                ),
            )

            print(
                "Next:",
                item.next,
            )

            print(
                "Node:",
                item.metadata.get("source"),
            )

            print(
                "Step:",
                item.metadata.get("step"),
            )

    finally:

        # ==================================
        # 10. Cleanup
        # ==================================

        retriever.close()

        llm_client.close()

        print("\nFinRAG resources closed.")


if __name__ == "__main__":
    main()

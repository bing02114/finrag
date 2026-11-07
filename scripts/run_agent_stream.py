
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from finrag.agent.graph import create_agent


def main():

    # ======================================
    # 1. SQLite Checkpointer
    # ======================================

    db_path = Path(
        "data/checkpoints/finrag.db"
    )

    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
        timeout=30,
    )

    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    # ======================================
    # 2. Create Agent
    # ======================================

    graph, retriever, llm_client = create_agent(
        checkpointer=checkpointer
    )

    query = (
        "变压器乙炔超标和基础沉降超标"
        "分别对应什么重大隐患条件？"
    )

    # Use a fresh Thread ID
    config = {
        "configurable": {
            "thread_id": "finrag-stream-demo"
        },
        "recursion_limit": 20,
    }

    initial_state = {
        "original_query": query,
        "current_query": query,
        "evidence_ids": [],
        "evidence": [],
        "grading_status": "",
        "covered_aspects": [],
        "missing_aspects": [],
        "retry_count": 0,
        "max_retries": 2,
        "seen_queries": [query],
        "history": [],
        "final_answer": "",
        "failure_reason": "",
    }

    try:

        print("\n========== FINRAG STREAMING ==========")

        # ==================================
        # 3. Stream Node Updates
        # ==================================

        for event in graph.stream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):

            for node_name, node_output in event.items():

                print(
                    f"\n[NODE COMPLETED] {node_name}"
                )

                if "evidence_ids" in node_output:
                    print(
                        "Evidence:",
                        node_output["evidence_ids"],
                    )

                if "grading_status" in node_output:
                    print(
                        "Grading:",
                        node_output["grading_status"],
                    )

                if "final_answer" in node_output:
                    print(
                        "Answer:",
                        node_output["final_answer"],
                    )

        # ==================================
        # 4. Final State
        # ==================================

        print("\n========== FINAL STATE ==========")

        snapshot = graph.get_state(config)

        print("Status:", snapshot.values["grading_status"])

        print("Evidence:", snapshot.values["evidence_ids"])

        print("Answer:")
        print(snapshot.values["final_answer"])

        print("Next:", snapshot.next)

    finally:

        retriever.close()
        llm_client.close()
        conn.close()


if __name__ == "__main__":
    main()

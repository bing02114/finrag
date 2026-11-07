
from langgraph.checkpoint.memory import InMemorySaver

from finrag.agent.graph import create_agent


def main():

    # ==================================
    # 1. Shared Checkpointer
    # ==================================

    checkpointer = InMemorySaver()

    graph, retriever, llm_client = create_agent(
        checkpointer=checkpointer
    )

    try:

        # ==================================
        # 2. Thread A
        # ==================================

        config_a = {
            "configurable": {
                "thread_id": "finrag-user-A"
            },
            "recursion_limit": 20,
        }

        query_a = "变压器乙炔含量达到多少属于重大隐患？"

        initial_a = {
            "original_query": query_a,
            "current_query": query_a,
            "evidence_ids": [],
            "evidence": [],
            "grading_status": "",
            "covered_aspects": [],
            "missing_aspects": [],
            "retry_count": 0,
            "max_retries": 2,
            "seen_queries": [query_a],
            "history": [],
            "final_answer": "",
            "failure_reason": "",
        }

        print("\n===== THREAD A =====")

        result_a = graph.invoke(
            initial_a,
            config=config_a,
        )

        print("Answer A:")
        print(result_a["final_answer"])

        # ==================================
        # 3. Thread B
        # ==================================

        config_b = {
            "configurable": {
                "thread_id": "finrag-user-B"
            },
            "recursion_limit": 20,
        }

        query_b = "汽轮机阀门卡涩属于什么重大隐患？"

        initial_b = {
            "original_query": query_b,
            "current_query": query_b,
            "evidence_ids": [],
            "evidence": [],
            "grading_status": "",
            "covered_aspects": [],
            "missing_aspects": [],
            "retry_count": 0,
            "max_retries": 2,
            "seen_queries": [query_b],
            "history": [],
            "final_answer": "",
            "failure_reason": "",
        }

        print("\n===== THREAD B =====")

        result_b = graph.invoke(
            initial_b,
            config=config_b,
        )

        print("Answer B:")
        print(result_b["final_answer"])

        # ==================================
        # 4. Inspect Thread States
        # ==================================

        state_a = graph.get_state(config_a)
        state_b = graph.get_state(config_b)

        print("\n===== ISOLATION CHECK =====")

        print("Thread A:", config_a["configurable"]["thread_id"])
        print("Thread A Query:", state_a.values["original_query"])

        print("\nThread B:", config_b["configurable"]["thread_id"])
        print("Thread B Query:", state_b.values["original_query"])

        # ==================================
        # 5. Assertions
        # ==================================

        assert (
            state_a.values["original_query"]
            != state_b.values["original_query"]
        )

        assert (
            state_a.values["evidence_ids"]
            != state_b.values["evidence_ids"]
        )

        assert state_a.next == ()
        assert state_b.next == ()

        print("\nThread isolation test passed!")

    finally:

        retriever.close()
        llm_client.close()


if __name__ == "__main__":
    main()

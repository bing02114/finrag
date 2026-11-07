
from finrag.retrieval.pipeline import HybridRetriever


def main():

    retriever = HybridRetriever(
        collection_name="power_safety_children",
        device="cuda",
        recall_k=20,
        candidate_k=20,
        final_k=5,
    )

    query = (
        "变压器乙炔超标和基础沉降超标"
        "分别对应什么重大隐患条件？"
    )

    try:

        result = retriever.retrieve(query)

        print("\n===== RETRIEVAL RESULT =====")

        print("Query:", result["query"])
        print("Union count:", result["union_count"])
        print("Candidate count:", result["candidate_count"])

        print("\n===== FINAL EVIDENCE =====")

        for rank, item in enumerate(
            result["results"], 1
        ):

            print(f"\nRank {rank}")
            print("ID:", item["id"])

            print(
                "Reranker:",
                round(item["reranker_score"], 4)
            )

            print(
                "RRF:",
                round(item["rrf_score"], 6)
            )

            print("Text:", item["text"])

        print(
            "\nLatency:",
            round(result["latency"], 4),
            "s",
        )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()

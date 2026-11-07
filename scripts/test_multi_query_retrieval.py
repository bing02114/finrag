
from finrag.retrieval.pipeline import HybridRetriever


TEST_CASES = [

    {
        "id": "H10",
        "query": (
            "变压器乙炔超标和基础沉降超标"
            "分别对应什么重大隐患条件？"
        ),
        "sub_queries": [
            "变压器乙炔超标对应什么重大隐患条件？",
            "变压器基础沉降超标对应什么重大隐患条件？",
        ],
        "gold": [
            "D01_第五条:5",
            "D01_第五条:6",
        ],
    },

    {
        "id": "H11",
        "query": (
            "新能源场站并网能力不足与发电机组"
            "未开展进相试验分别有什么判定要求？"
        ),
        "sub_queries": [
            "新能源场站并网能力不足的判定要求是什么？",
            "发电机组未开展进相试验的判定要求是什么？",
        ],
        "gold": [
            "D01_第五条:4",
            "D01_第五条:3",
        ],
    },

    {
        "id": "H12",
        "query": (
            "汽轮机阀门卡涩与缺少转速飞升"
            "应急措施分别如何判定？"
        ),
        "sub_queries": [
            "汽轮机阀门卡涩如何判定？",
            "汽轮机缺少转速飞升应急措施如何判定？",
        ],
        "gold": [
            "D01_第六条:0",
            "D01_第六条:1",
        ],
    },

]


def evaluate(result, gold):

    retrieved = set(result["evidence_ids"])
    gold_set = set(gold)

    hits = retrieved & gold_set

    recall = len(hits) / len(gold_set)

    all_hit = int(
        gold_set.issubset(retrieved)
    )

    hit1 = int(
        bool(result["evidence_ids"])
        and result["evidence_ids"][0] in gold_set
    )

    return {
        "hit1": hit1,
        "recall": recall,
        "all_hit": all_hit,
        "hits": sorted(hits),
    }


def main():

    retriever = HybridRetriever(
        collection_name="power_safety_children",
        device="cuda",
        recall_k=20,
        candidate_k=20,
        final_k=10,
    )

    baseline_metrics = []
    multi_metrics = []

    try:

        for case in TEST_CASES:

            print("\n" + "=" * 65)

            print("TEST:", case["id"])

            print("Original:", case["query"])

            # ==================================
            # 1. Baseline
            # ==================================

            print("\n===== BASELINE =====")

            baseline = retriever.retrieve(
                case["query"]
            )

            baseline_eval = evaluate(
                baseline,
                case["gold"],
            )

            print(
                "Evidence:",
                baseline["evidence_ids"],
            )

            print(
                "Metrics:",
                baseline_eval,
            )

            baseline_metrics.append(
                baseline_eval
            )

            # ==================================
            # 2. Multi-query
            # ==================================

            print("\n===== MULTI-QUERY =====")

            multi = retriever.retrieve_many(
                case["sub_queries"],
                final_k=10,
            )

            multi_eval = evaluate(
                multi,
                case["gold"],
            )

            print(
                "Evidence:",
                multi["evidence_ids"],
            )

            print(
                "Metrics:",
                multi_eval,
            )

            print("\nEvidence Provenance:")

            for item in multi["results"]:

                print(
                    item["id"],
                    "RRF:",
                    round(item["rrf_score"], 6),
                    "Ranks:",
                    item["query_ranks"],
                )

            multi_metrics.append(
                multi_eval
            )

        # ======================================
        # 3. Aggregate Metrics
        # ======================================

        def summarize(metrics):

            n = len(metrics)

            return {
                "Hit@1": sum(
                    x["hit1"] for x in metrics
                ) / n,

                "Recall@10": sum(
                    x["recall"] for x in metrics
                ) / n,

                "AllHit@10": sum(
                    x["all_hit"] for x in metrics
                ) / n,
            }

        print("\n" + "=" * 65)

        print("===== FINAL COMPARISON =====")

        print(
            "Baseline:",
            summarize(baseline_metrics),
        )

        print(
            "Multi-query:",
            summarize(multi_metrics),
        )

    finally:

        retriever.close()


if __name__ == "__main__":
    main()

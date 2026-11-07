
import time
import statistics

from finrag.retrieval.pipeline import HybridRetriever


# ==========================================
# 1. Benchmark
# ==========================================

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


TOP_K = 10


# ==========================================
# 2. Evaluation Metrics
# ==========================================

def evaluate(results, gold):

    gold_set = set(gold)

    retrieved = [
        item["id"]
        for item in results
    ]

    retrieved_set = set(retrieved)

    hits = retrieved_set & gold_set

    recall = len(hits) / len(gold_set)

    precision = len(hits) / max(len(retrieved), 1)

    all_hit = int(
        gold_set.issubset(retrieved_set)
    )

    hit1 = int(
        bool(retrieved)
        and retrieved[0] in gold_set
    )

    first_gold_rank = next(
        (
            rank
            for rank, doc_id in enumerate(
                retrieved, 1
            )
            if doc_id in gold_set
        ),
        None,
    )

    mrr = (
        1.0 / first_gold_rank
        if first_gold_rank is not None
        else 0.0
    )

    return {
        "hit1": hit1,
        "recall": recall,
        "precision": precision,
        "all_hit": all_hit,
        "mrr": mrr,
        "hits": sorted(hits),
        "first_gold_rank": first_gold_rank,
    }


# ==========================================
# 3. Global Reranking
# ==========================================

def global_rerank(
    retriever,
    original_query,
    sub_queries,
    top_k=10,
):

    start = time.perf_counter()

    # ======================================
    # Step 1: Independent Retrieval
    # ======================================

    candidate_map = {}

    per_query_info = []

    for query_index, query in enumerate(
        sub_queries, 1
    ):

        print(
            f"\n[Global Rerank] "
            f"Retrieval {query_index}/{len(sub_queries)}"
        )

        print("Query:", query)

        result = retriever.retrieve(query)

        per_query_info.append({
            "query": query,
            "latency": result["latency"],
            "evidence_ids": result["evidence_ids"],
        })

        # Collect per-query Top-K
        for rank, item in enumerate(
            result["results"], 1
        ):

            doc_id = item["id"]

            if doc_id not in candidate_map:

                candidate_map[doc_id] = {
                    "id": doc_id,
                    "rrf_score": 0.0,
                    "source_queries": [],
                    "query_ranks": {},
                }

            candidate = candidate_map[doc_id]

            candidate["source_queries"].append(
                query
            )

            candidate["query_ranks"][query] = rank

    # ======================================
    # Step 2: Union + Deduplication
    # ======================================

    candidates = list(
        candidate_map.values()
    )

    print("\n[Global Rerank]")

    print(
        "Unique candidates:",
        len(candidates),
    )

    # ======================================
    # Step 3: Global Cross-Encoder
    # ======================================

    rerank_start = time.perf_counter()

    # Reuse the existing CrossEncoder instance.
    # Only the query changes to the original query.

    final_results = retriever._rerank(
        original_query,
        candidates,
    )

    rerank_latency = (
        time.perf_counter() - rerank_start
    )

    # ======================================
    # Step 4: Final Top-K
    # ======================================

    final_results = final_results[:top_k]

    total_latency = (
        time.perf_counter() - start
    )

    return {
        "results": final_results,
        "evidence_ids": [
            item["id"]
            for item in final_results
        ],
        "candidate_count": len(candidates),
        "per_query_info": per_query_info,
        "rerank_latency": rerank_latency,
        "total_latency": total_latency,
    }


# ==========================================
# 4. Display Results
# ==========================================

def print_result(
    name,
    result,
    metrics,
):

    print(f"\n===== {name} =====")

    print(
        "Evidence:",
        result["evidence_ids"],
    )

    print(
        "Metrics:",
        metrics,
    )

    print(
        "Latency:",
        round(result["total_latency"], 4),
        "s",
    )

    if "rerank_latency" in result:

        print(
            "Global Reranker Latency:",
            round(
                result["rerank_latency"],
                4,
            ),
            "s",
        )


# ==========================================
# 5. Main
# ==========================================

def main():

    retriever = HybridRetriever(
        collection_name="power_safety_children",
        device="cuda",
        recall_k=20,
        candidate_k=20,
        final_k=TOP_K,
    )

    summaries = {
        "baseline": [],
        "rrf": [],
        "global": [],
    }

    try:

        for case in TEST_CASES:

            print("\n" + "=" * 70)

            print("TEST:", case["id"])

            print("Original:", case["query"])

            # ==================================
            # A. Baseline
            # ==================================

            baseline_start = time.perf_counter()

            baseline = retriever.retrieve(
                case["query"]
            )

            baseline_elapsed = (
                time.perf_counter()
                - baseline_start
            )

            baseline_result = {
                **baseline,
                "total_latency": baseline_elapsed,
            }

            baseline_metrics = evaluate(
                baseline["results"],
                case["gold"],
            )

            print_result(
                "BASELINE",
                baseline_result,
                baseline_metrics,
            )

            summaries["baseline"].append(
                baseline_metrics
            )

            # ==================================
            # B. Multi-query RRF
            # ==================================

            rrf_start = time.perf_counter()

            rrf_result = retriever.retrieve_many(
                case["sub_queries"],
                final_k=TOP_K,
            )

            rrf_elapsed = (
                time.perf_counter()
                - rrf_start
            )

            rrf_result["total_latency"] = (
                rrf_elapsed
            )

            rrf_metrics = evaluate(
                rrf_result["results"],
                case["gold"],
            )

            print_result(
                "MULTI-QUERY RRF",
                rrf_result,
                rrf_metrics,
            )

            summaries["rrf"].append(
                rrf_metrics
            )

            # ==================================
            # C. Global Reranking
            # ==================================

            global_result = global_rerank(
                retriever=retriever,
                original_query=case["query"],
                sub_queries=case["sub_queries"],
                top_k=TOP_K,
            )

            global_metrics = evaluate(
                global_result["results"],
                case["gold"],
            )

            print_result(
                "GLOBAL RERANK",
                global_result,
                global_metrics,
            )

            summaries["global"].append(
                global_metrics
            )

            # ==================================
            # D. Evidence Comparison
            # ==================================

            print("\n===== GOLD RANK COMPARISON =====")

            for gold_id in case["gold"]:

                def get_rank(result):

                    ids = result["evidence_ids"]

                    return (
                        ids.index(gold_id) + 1
                        if gold_id in ids
                        else None
                    )

                print(
                    gold_id,
                    "| Baseline:",
                    get_rank(baseline_result),
                    "| RRF:",
                    get_rank(rrf_result),
                    "| Global:",
                    get_rank(global_result),
                )

    finally:

        retriever.close()

    # ======================================
    # 6. Final Summary
    # ======================================

    print("\n" + "=" * 70)

    print("========== FINAL SUMMARY ==========")

    def summarize(metrics):

        n = len(metrics)

        keys = [
            "hit1",
            "recall",
            "precision",
            "all_hit",
            "mrr",
        ]

        return {
            key: round(
                sum(item[key] for item in metrics) / n,
                4,
            )
            for key in keys
        }

    for name, metrics in summaries.items():

        print(
            name.upper(),
            summarize(metrics),
        )


if __name__ == "__main__":
    main()

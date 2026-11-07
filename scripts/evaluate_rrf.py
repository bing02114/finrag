
import json
from pathlib import Path
from statistics import mean

INPUT = Path(
    "experiments/retrieval/hard_query_results.json"
)

OUTPUT = Path(
    "experiments/retrieval/rrf_results.json"
)

RRF_K = 60
TOP_K = 5

data = json.loads(INPUT.read_text(encoding="utf-8"))


# 1. RRF fusion
def rrf_fusion(bm25_ids, dense_ids, k=60):

    scores = {}

    for ranked_list in [bm25_ids, dense_ids]:
        for rank, doc_id in enumerate(ranked_list, 1):
            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], doc_id)
    )


# 2. Evaluation
def evaluate(ranked_ids, gold_ids, k=5):
    retrieved = ranked_ids[:k]
    gold = set(gold_ids)

    matched = gold.intersection(retrieved)

    recall = len(matched) / len(gold)
    all_hit = int(gold.issubset(set(retrieved)))

    ranks = [
        rank
        for rank, doc_id in enumerate(retrieved, 1)
        if doc_id in gold
    ]

    rr = 1 / min(ranks) if ranks else 0.0

    return {
        "recall": recall,
        "all_hit": all_hit,
        "rr": rr,
    }


# 3. Compare three methods
results = []

for row in data:
    bm25_ids = row["bm25"]
    dense_ids = row["dense"]
    gold = row["gold"]

    hybrid_ids = rrf_fusion(
        bm25_ids,
        dense_ids,
        k=RRF_K,
    )

    result = {
        "id": row["id"],
        "category": row["category"],
        "gold": gold,
        "bm25": bm25_ids,
        "dense": dense_ids,
        "hybrid": hybrid_ids[:TOP_K],
    }

    result["metrics"] = {
        "bm25": evaluate(bm25_ids, gold, TOP_K),
        "dense": evaluate(dense_ids, gold, TOP_K),
        "hybrid": evaluate(hybrid_ids, gold, TOP_K),
    }

    results.append(result)

    print(f'\n===== {row["id"]} =====')
    print("Gold:", gold)
    print("BM25:", bm25_ids[:3])
    print("Dense:", dense_ids[:3])
    print("Hybrid:", hybrid_ids[:3])

    print(
        "Evidence Recall@5:",
        {m: result["metrics"][m]["recall"]
         for m in ["bm25", "dense", "hybrid"]}
    )


# 4. Overall and per-category metrics
def summarize(rows):
    for method in ["bm25", "dense", "hybrid"]:
        metrics = [r["metrics"][method] for r in rows]

        print(
            f"{method:7s} "
            f"Recall@5={mean(m['recall'] for m in metrics):.3f} "
            f"AllHit@5={mean(m['all_hit'] for m in metrics):.3f} "
            f"MRR@5={mean(m['rr'] for m in metrics):.3f}"
        )


print("\n===== OVERALL SUMMARY =====")
summarize(results)

print("\n===== CATEGORY SUMMARY =====")

for category in sorted(set(r["category"] for r in results)):
    subset = [
        r for r in results
        if r["category"] == category
    ]

    print(f"\n{category} (n={len(subset)})")
    summarize(subset)


# 5. Save results
OUTPUT.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nSaved:", OUTPUT)

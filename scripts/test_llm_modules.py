
from finrag.llm.grader import EvidenceGrader
from finrag.llm.rewriter import QueryRewriter


def main():

    grader = EvidenceGrader()
    rewriter = QueryRewriter()

    # =================================
    # Test 1: Sufficient Evidence
    # =================================

    query = (
        "变压器乙炔超标和基础沉降超标"
        "分别对应什么重大隐患条件？"
    )

    evidence = [
        {
            "id": "D01_第五条:5",
            "text": (
                "（六）直流±800千伏、交流1000千伏"
                "以上变压器监测的乙炔含量达到5微升/升"
                "或周增量超过2微升/升。"
            ),
        },
        {
            "id": "D01_第五条:6",
            "text": (
                "（七）500千伏以上变电站变压器"
                "基础沉降差或倾斜超过0.003L。"
            ),
        },
    ]

    print("\n===== GRADER TEST =====")

    grade = grader.grade(
        query,
        evidence,
    )

    print("Status:", grade.status)
    print("Covered:", grade.covered_aspects)
    print("Missing:", grade.missing_aspects)
    print("Reason:", grade.reason)

    assert grade.status == "sufficient"

    # =================================
    # Test 2: Query Rewrite
    # =================================

    print("\n===== REWRITER TEST =====")

    missing = [
        "变压器基础沉降差或倾斜的判定标准"
    ]

    rewritten = rewriter.rewrite(
        query,
        missing,
    )

    print("Original:", query)
    print("Rewritten:", rewritten)

    assert rewritten.strip()

    print("\nAll LLM module tests passed!")


if __name__ == "__main__":
    main()

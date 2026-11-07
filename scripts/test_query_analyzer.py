
from finrag.llm.analyzer import QueryAnalyzer


def main():

    analyzer = QueryAnalyzer()

    test_cases = [

        {
            "name": "Simple Query",
            "query": (
                "变压器乙炔含量达到多少属于重大隐患？"
            ),
            "context": "",
            "expected": "direct",
        },

        {
            "name": "Multi-part Query",
            "query": (
                "变压器乙炔超标和基础沉降超标"
                "分别对应什么重大隐患条件？"
            ),
            "context": "",
            "expected": "decompose",
        },

        {
            "name": "Complex Multi-part Query",
            "query": (
                "汽轮机阀门卡涩与缺少转速飞升"
                "应急措施分别如何判定？"
            ),
            "context": "",
            "expected": "decompose",
        },

        {
            "name": "Ambiguous Query",
            "query": (
                "这个标准适用于所有变压器吗？"
            ),
            "context": "",
            "expected": "clarify",
        },

        {
            "name": "Context-aware Query",
            "query": (
                "这个标准适用于所有变压器吗？"
            ),
            "context": (
                "上一轮讨论的是《电力重大事故隐患"
                "判定标准》第五条第（六）项，"
                "涉及乙炔含量达到5微升/升的条件。"
            ),
            "expected": "direct",
        },
    ]

    correct = 0

    print("\n========== QUERY ANALYZER ==========")

    for item in test_cases:

        print("\n" + "=" * 60)

        print("Test:", item["name"])

        print("Query:", item["query"])

        result = analyzer.analyze(
            query=item["query"],
            conversation_context=item["context"],
        )

        print("\nStrategy:", result.strategy)

        print("Retrieval Query:", result.retrieval_query)
        
        print(
            "Clarification:",
            result.clarification_question,
        )

        print(
            "Clarification:",
            result.clarification_question,
        )

        print("Reasoning:", result.reasoning)

        passed = (
            result.strategy == item["expected"]
        )

        if passed:
            correct += 1

        print(
            "Expected:",
            item["expected"],
        )

        print(
            "Result:",
            "PASS" if passed else "FAIL",
        )

    print("\n========== SUMMARY ==========")

    print("Total:", len(test_cases))

    print("Correct:", correct)

    print(
        "Accuracy:",
        f"{correct / len(test_cases):.2%}",
    )


if __name__ == "__main__":
    main()

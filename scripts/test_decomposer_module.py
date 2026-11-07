
from finrag.llm.decomposer import QueryDecomposer


def main():

    decomposer = QueryDecomposer()

    test_cases = [

        {
            "id": "H10",
            "query": (
                "变压器乙炔超标和基础沉降超标"
                "分别对应什么重大隐患条件？"
            ),
            "expected_count": 2,
        },

        {
            "id": "H11",
            "query": (
                "新能源场站并网能力不足与发电机组"
                "未开展进相试验分别有什么判定要求？"
            ),
            "expected_count": 2,
        },

        {
            "id": "H12",
            "query": (
                "汽轮机阀门卡涩与缺少转速飞升"
                "应急措施分别如何判定？"
            ),
            "expected_count": 2,
        },

    ]

    print("\n========== DECOMPOSER MODULE TEST ==========")

    for item in test_cases:

        print("\n" + "=" * 60)

        print("ID:", item["id"])

        print("Original:", item["query"])

        result = decomposer.decompose(
            item["query"]
        )

        print("\nSub-queries:")

        for index, query in enumerate(
            result["queries"], 1
        ):

            print(f"{index}. {query}")

        print("\nQuery count:", len(result["queries"]))

        print("Latency:", result["latency"], "s")

        print("Usage:", result["usage"])

        assert (
            len(result["queries"])
            == item["expected_count"]
        )

        assert (
            len(result["queries"])
            == len(set(result["queries"]))
        )

        print("Result: PASS")

    print("\nAll Decomposer tests passed!")


if __name__ == "__main__":
    main()

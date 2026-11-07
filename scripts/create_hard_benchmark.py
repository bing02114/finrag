import json
from pathlib import Path

OUTPUT = Path(
    "experiments/retrieval/benchmarks/"
    "hard_queries_v1.jsonl"
)

def item(article, index):
    return f"D01_{article}:{index}"

questions = [
    # Exact terms
    ("H01", "exact_term",
     "直流±800千伏变压器乙炔含量达到多少微升/升属于重大隐患？",
     [item("第五条", 5)]),

    ("H02", "exact_term",
     "220千伏以上新能源场站的低电压穿越能力有什么要求？",
     [item("第五条", 4)]),

    ("H03", "exact_term",
     "500千伏以上变电站的GIS设备基础沉降量超过多少毫米属于重大隐患？",
     [item("第五条", 6)]),

    # Paraphrase
    ("H04", "paraphrase",
     "风力发电站遇到电网电压骤降时无法维持并网运行，会被认定为严重安全隐患吗？",
     [item("第五条", 4)]),

    ("H05", "paraphrase",
     "大型发电机没有按照规定进行无功吸收能力测试，是否违反隐患判定标准？",
     [item("第五条", 3)]),

    ("H06", "paraphrase",
     "汽轮机主汽阀动作不顺畅，可能卡住，这种情况是否属于重大隐患？",
     [item("第六条", 0)]),

    # Hard negatives
    ("H07", "hard_negative",
     "变压器基础倾斜超过标准，而不是气体含量异常，应查哪一项？",
     [item("第五条", 6)]),

    ("H08", "hard_negative",
     "发电机组未配置电力系统稳定器，与未开展进相试验，前者属于哪一项？",
     [item("第五条", 2)]),

    ("H09", "hard_negative",
     "汽轮机转速飞升的应急措施缺失属于哪项隐患？",
     [item("第六条", 1)]),

    # Multiple evidence
    ("H10", "multi_evidence",
     "变压器乙炔超标和基础沉降超标分别对应什么重大隐患条件？",
     [item("第五条", 5), item("第五条", 6)]),

    ("H11", "multi_evidence",
     "新能源场站并网能力不足与发电机组未开展进相试验分别有什么判定要求？",
     [item("第五条", 4), item("第五条", 3)]),

    ("H12", "multi_evidence",
     "汽轮机阀门卡涩与缺少转速飞升应急措施分别如何判定？",
     [item("第六条", 0), item("第六条", 1)]),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    for qid, category, query, gold in questions:
        record = {
            "id": qid,
            "query": query,
            "category": category,
            "gold_children": gold,
            "verification_status": "pending"
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Created:", OUTPUT)
print("Questions:", len(questions))
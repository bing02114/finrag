
import time
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda"

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print("Model:", MODEL_NAME)
print("Dimension:", model.get_embedding_dimension())
print("GPU:", torch.cuda.get_device_name(0))

# Prepare a larger test dataset
base_texts = [
    "贵州茅台2024年营业收入为1741.44亿元。",
    "贵州茅台2023年营业收入为1476.94亿元。",
    "五粮液是一家中国白酒上市公司。",
    "公司的研发投入持续增长。",
    "营业收入和净利润是重要财务指标。",
    "经营性现金流反映企业经营活动的现金状况。",
    "公司的资产负债率反映财务杠杆水平。",
    "毛利率下降可能与原材料成本上涨有关。",
]

texts = base_texts * 125

batch_sizes = [4, 8, 16, 32, 64]

results = []

for batch_size in batch_sizes:

    print(f"\nTesting batch_size={batch_size}")

    # Warm-up
    model.encode(
        texts[:min(batch_size, len(texts))],
        batch_size=batch_size,
        convert_to_tensor=True
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    throughput = len(texts) / elapsed

    results.append({
        "batch_size": batch_size,
        "elapsed": elapsed,
        "throughput": throughput,
        "peak_memory": peak_memory
    })

    print("Elapsed:", round(elapsed, 3), "s")
    print("Throughput:", round(throughput, 2), "texts/s")
    print("Peak Memory:", round(peak_memory, 3), "GB")

print("\n===== FINAL RESULTS =====")

for result in results:
    print(result)

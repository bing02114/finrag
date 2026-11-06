
import torch
import time
from sentence_transformers import SentenceTransformer

# 1. Load model
model_name = "BAAI/bge-m3"

print(f"Loading model: {model_name}")

model = SentenceTransformer(
    model_name,
    device="cuda"
)

# 2. Check model information
print("\n===== MODEL INFO =====")
print("Model:", model_name)
print("Embedding dimension:",
      model.get_sentence_embedding_dimension())
print("Device:", model.device)
print("GPU:", torch.cuda.get_device_name(0))

# 3. Prepare test documents
texts = [
    "贵州茅台2024年营业收入为1741.44亿元。",
    "贵州茅台2023年营业收入为1476.94亿元。",
    "五粮液是一家中国白酒上市公司。",
    "人工智能正在改变金融行业。",
    "Retrieval Augmented Generation combines retrieval and generation."
]

# 4. Batch embedding
torch.cuda.reset_peak_memory_stats()

start_time = time.perf_counter()

embeddings = model.encode(
    texts,
    batch_size=4,
    normalize_embeddings=True,
    convert_to_tensor=True,
    show_progress_bar=True
)

elapsed = time.perf_counter() - start_time

# 5. Check embedding results
print("\n===== EMBEDDING RESULTS =====")
print("Shape:", embeddings.shape)
print("Elapsed:", round(elapsed, 4), "seconds")
print("Throughput:", round(len(texts) / elapsed, 2), "texts/sec")

print(
    "Peak GPU Memory:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    "GB"
)

print(
    "Vector norm:",
    embeddings[0].norm().item()
)

# 6. Calculate cosine similarity
similarities = embeddings @ embeddings.T

print("\n===== COSINE SIMILARITY =====")

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print(
            f"Text {i} vs Text {j}: "
            f"{similarities[i][j].item():.4f}"
        )

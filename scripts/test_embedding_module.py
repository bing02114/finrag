
from finrag.embedding.model import EmbeddingModel

# Initialize
embedding = EmbeddingModel(
    model_name="BAAI/bge-m3",
    device="cuda",
    batch_size=64,
)

# Test queries
queries = [
    "贵州茅台2024年营业收入是多少？",
    "公司的研发投入情况如何？",
]

# Test documents
documents = [
    "贵州茅台2024年实现营业收入1741.44亿元。",
    "公司持续增加研发投入，推动技术创新。",
    "五粮液是中国知名白酒企业。",
]

# Encode
query_vectors = [
    embedding.encode_query(query)
    for query in queries
]

document_vectors = embedding.encode_documents(documents)

# Validate
print("\n===== VALIDATION =====")

print("Dimension:", embedding.get_dimension())
print("Query shape:", len(query_vectors), len(query_vectors[0]))
print("Document shape:", len(document_vectors), len(document_vectors[0]))

assert len(query_vectors[0]) == 1024
assert len(document_vectors[0]) == 1024

print("Embedding module works!")

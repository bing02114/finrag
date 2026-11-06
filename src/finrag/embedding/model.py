
import time
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        max_seq_length: Optional[int] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Automatically select device
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Load model
        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        # Optionally control maximum sequence length
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

        self.dimension = self.model.get_embedding_dimension()

        print(f"Loaded embedding model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Dimension: {self.dimension}")

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip()
               for text in texts):
            raise ValueError("Texts must contain non-empty strings.")

        start_time = time.perf_counter()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        elapsed = time.perf_counter() - start_time

        print(
            f"Encoded {len(texts)} texts in {elapsed:.4f}s "
            f"({len(texts) / elapsed:.2f} texts/s)"
        )

        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        return self.encode([query])[0]

    def encode_documents(
        self,
        documents: List[str]
    ) -> List[List[float]]:
        return self.encode(documents)

    def get_dimension(self) -> int:
        return self.dimension

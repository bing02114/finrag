
import re
import time
from typing import Any

import jieba
import numpy as np
import torch

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from finrag.embedding.model import EmbeddingModel


class HybridRetriever:

    def __init__(
        self,
        collection_name="power_safety_children",
        qdrant_url="http://127.0.0.1:6333",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        device="cuda",
        recall_k=20,
        candidate_k=20,
        final_k=5,
        rrf_k=60,
        batch_size=8,
    ):

        self.collection_name = collection_name
        self.recall_k = recall_k
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.rrf_k = rrf_k
        self.batch_size = batch_size

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = device

        # Qdrant
        self.client = QdrantClient(url=qdrant_url)

        # Embedding
        self.embedding = EmbeddingModel(
            model_name=embedding_model,
            device=device,
            batch_size=32,
        )

        # Reranker
        self.reranker = CrossEncoder(
            reranker_model,
            device=device,
            max_length=512,
            activation_fn=torch.nn.Identity(),
        )

        # Load corpus
        self.corpus = {}
        self.corpus_ids = []

        self._load_corpus()
        self._initialize_bm25()

        print("\nHybridRetriever initialized.")
        print("Collection:", collection_name)
        print("Documents:", len(self.corpus))
        print("Device:", self.device)

    # ==================================
    # 1. Load Corpus
    # ==================================

    @staticmethod
    def _get_id(payload):

        return (
            f'{payload["parent_id"]}:'
            f'{payload["child_index"]}'
        )

    def _load_corpus(self):

        records = []
        offset = None

        while True:

            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            records.extend(batch)

            if offset is None:
                break

        for record in records:

            payload = record.payload
            doc_id = self._get_id(payload)

            self.corpus[doc_id] = {
                "text": payload["text"],
                "payload": payload,
            }

        self.corpus_ids = sorted(self.corpus)

        if not self.corpus:
            raise RuntimeError(
                "Qdrant collection is empty."
            )

    # ==================================
    # 2. BM25
    # ==================================

    @staticmethod
    def _tokenize(text):

        return [
            token
            for token in jieba.lcut(text.lower())
            if token.strip()
            and re.search(
                r"[\u4e00-\u9fffA-Za-z0-9]",
                token,
            )
        ]

    def _initialize_bm25(self):

        domain_terms = [
            "重大隐患",
            "重大事故隐患",
            "乙炔含量",
            "基础沉降",
            "进相试验",
            "电力系统稳定器",
            "低电压穿越",
            "汽轮机",
            "转速飞升",
            "高温高压汽水系统",
        ]

        for term in domain_terms:
            jieba.add_word(term)

        tokenized_corpus = [
            self._tokenize(
                self.corpus[doc_id]["text"]
            )
            for doc_id in self.corpus_ids
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=1.5,
            b=0.75,
        )

    def _search_bm25(self, query):

        scores = self.bm25.get_scores(
            self._tokenize(query)
        )

        indices = np.argsort(
            -scores,
            kind="stable",
        )[:self.recall_k]

        return [
            {
                "id": self.corpus_ids[int(index)],
                "score": float(scores[index]),
                "rank": rank,
            }
            for rank, index in enumerate(indices, 1)
        ]

    # ==================================
    # 3. Dense Retrieval
    # ==================================

    def _search_dense(self, query):

        query_vector = self.embedding.encode_query(
            query
        )

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.recall_k,
            with_payload=True,
        ).points

        return [
            {
                "id": self._get_id(hit.payload),
                "score": float(hit.score),
                "rank": rank,
            }
            for rank, hit in enumerate(hits, 1)
        ]

    # ==================================
    # 4. RRF Fusion
    # ==================================

    def _rrf_fusion(
        self,
        bm25_results,
        dense_results,
    ):

        scores = {}

        for results in [
            bm25_results,
            dense_results,
        ]:

            for item in results:

                doc_id = item["id"]
                rank = item["rank"]

                scores[doc_id] = (
                    scores.get(doc_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        return [
            {
                "id": doc_id,
                "rrf_score": score,
            }
            for doc_id, score in ranked
        ]

    # ==================================
    # 5. Cross-Encoder Reranking
    # ==================================

    def _rerank(self, query, candidates):

        if not candidates:
            return []

        pairs = [
            (
                query,
                self.corpus[item["id"]]["text"],
            )
            for item in candidates
        ]

        scores = self.reranker.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: -float(item[1]),
        )

        return [
            {
                "id": item["id"],
                "reranker_score": float(score),
                "rrf_score": item["rrf_score"],
                "text": self.corpus[item["id"]]["text"],
                "payload": self.corpus[item["id"]]["payload"],
            }
            for item, score in ranked[:self.final_k]
        ]

    # ==================================
    # 6. Public Interface
    # ==================================

    def retrieve(self, query: str) -> dict[str, Any]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        start = time.perf_counter()

        # BM25
        bm25_results = self._search_bm25(query)

        # Dense
        dense_results = self._search_dense(query)

        # Candidate union
        union_ids = {
            item["id"]
            for item in bm25_results + dense_results
        }

        # RRF
        fused = self._rrf_fusion(
            bm25_results,
            dense_results,
        )

        candidates = fused[:self.candidate_k]

        # Reranker
        final_results = self._rerank(
            query,
            candidates,
        )

        elapsed = time.perf_counter() - start

        return {
            "query": query,
            "results": final_results,
            "evidence_ids": [
                item["id"]
                for item in final_results
            ],
            "candidate_ids": [
                item["id"]
                for item in candidates
            ],
            "union_count": len(union_ids),
            "candidate_count": len(candidates),
            "bm25_results": bm25_results,
            "dense_results": dense_results,
            "rrf_results": candidates,
            "latency": elapsed,
        }

    
    # ======================================
    # 7. Multi-query Retrieval
    # ======================================

    def retrieve_many(
        self,
        queries: list[str],
        final_k: int = 10,
    ) -> dict[str, Any]:
        """
        Execute independent Hybrid Retrieval
        for multiple sub-queries.

        Merge ranked results using RRF,
        then deduplicate by Child ID.

        No additional global reranking is performed.
        """

        if not queries:
            raise ValueError(
                "Queries cannot be empty."
            )

        if final_k < 1:
            raise ValueError(
                "final_k must be positive."
            )

        # Remove empty queries and duplicates
        queries = list(
            dict.fromkeys(
                q.strip()
                for q in queries
                if q and q.strip()
            )
        )

        if not queries:
            raise ValueError(
                "No valid queries."
            )

        start = time.perf_counter()

        # Evidence ID -> merged information
        merged = {}

        per_query_results = []

        # ==================================
        # 1. Independent Retrieval
        # ==================================

        for query_index, query in enumerate(
            queries, 1
        ):

            print(
                f"\n[Multi-query Retrieval] "
                f"Query {query_index}/{len(queries)}"
            )

            print("Query:", query)

            result = self.retrieve(query)

            per_query_results.append({
                "query": query,
                "evidence_ids": result["evidence_ids"],
                "latency": result["latency"],
            })

            # ==================================
            # 2. Collect Ranked Evidence
            # ==================================

            for rank, item in enumerate(
                result["results"], 1
            ):

                doc_id = item["id"]

                if doc_id not in merged:

                    merged[doc_id] = {
                        "id": doc_id,
                        "text": item["text"],
                        "payload": item["payload"],
                        "rrf_score": 0.0,
                        "source_queries": [],
                        "query_ranks": {},
                        "per_query_scores": {},
                    }

                current = merged[doc_id]

                # RRF based on per-query reranker rank
                current["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                current["source_queries"].append(
                    query
                )

                current["query_ranks"][query] = rank

                current["per_query_scores"][query] = (
                    item["reranker_score"]
                )

        # ==================================
        # 3. Sort by Multi-query RRF
        # ==================================

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["id"],
            ),
        )

        # ==================================
        # 4. Deduplication
        # ==================================

        final_results = ranked[:final_k]

        evidence_ids = [
            item["id"]
            for item in final_results
        ]

        # ==================================
        # 5. Statistics
        # ==================================

        elapsed = time.perf_counter() - start

        print("\n[Multi-query Fusion]")

        print("Sub-query count:", len(queries))

        print("Unique evidence:", len(merged))

        print("Final evidence:", len(final_results))

        print("Latency:", round(elapsed, 4), "s")

        return {
            "queries": queries,
            "results": final_results,
            "evidence_ids": evidence_ids,
            "candidate_count": len(merged),
            "per_query_results": per_query_results,
            "latency": elapsed,
        }

    def close(self):
        self.client.close()
